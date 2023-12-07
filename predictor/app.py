import importlib.util
import json
import os
import shutil
import time
import uuid

from orthogonalizer import othogonalize_poly

from .downloader import download
from .prediction import run_prediction
from .raster2polygon import polygonizer
from .vectorizer import vectorize


def predict(
    bbox,
    model_path,
    zoom_level,
    tms_url,
    tile_size=256,
    base_path=None,
    confidence=0.5,
    area_threshold=3,
    tolerance=0.5,
    tile_overlap_distance=0.15,
    merge_adjancent_polygons=True,
    use_raster2polygon=False,
    remove_metadata=True,
    use_josm_q=False,
    max_angle_change=15,
    skew_tolerance=15,
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
        tile_overlap_distance : Provides tile overlap distance to remove the strip between predictions, Defaults to 0.15m
        merge_adjancent_polygons(bool,optional) : Merges adjacent self intersecting or containing each other polygons
    """
    if base_path:
        base_path = os.path.join(base_path, "prediction", str(uuid.uuid4()))
    else:
        base_path = os.path.join(os.getcwd(), "prediction", str(uuid.uuid4()))

    os.makedirs(base_path, exist_ok=True)
    download_path = os.path.join(base_path, "image")
    os.makedirs(download_path, exist_ok=True)

    image_download_path = download(
        bbox,
        zoom_level=zoom_level,
        tms_url=tms_url,
        tile_size=tile_size,
        download_path=download_path,
    )

    prediction_path = os.path.join(base_path, "prediction")
    os.makedirs(prediction_path, exist_ok=True)

    prediction_path = run_prediction(
        model_path,
        image_download_path,
        prediction_path=prediction_path,
        confidence=confidence,
        tile_overlap_distance=tile_overlap_distance,
    )
    start = time.time()

    geojson_path = os.path.join(base_path, "geojson")
    os.makedirs(geojson_path, exist_ok=True)
    geojson_path = os.path.join(geojson_path, "prediction.geojson")

    if use_raster2polygon:
        try:
            importlib.util.find_spec("raster2polygon")
        except ImportError:
            raise ImportError(
                "Raster2polygon is not installed. Install using pip install raster2polygon"
            )

        geojson_path = polygonizer(prediction_path, output_path=geojson_path)
    else:
        geojson_path = vectorize(
            prediction_path,
            output_path=geojson_path,
            area_threshold=area_threshold,
            tolerance=tolerance,
            merge_adjancent_polygons=merge_adjancent_polygons,
        )
    print(f"It took {round(time.time()-start)} sec to extract polygons")
    with open(geojson_path, "r") as f:
        prediction_geojson_data = json.load(f)
    if remove_metadata:
        shutil.rmtree(base_path)
    for feature in prediction_geojson_data["features"]:
        feature["properties"]["building"] = "yes"
        feature["properties"]["source"] = "fAIr"
        if use_josm_q is True:
            feature["geometry"] = othogonalize_poly(
                feature["geometry"],
                maxAngleChange=max_angle_change,
                skewTolerance=skew_tolerance,
            )
    return prediction_geojson_data
