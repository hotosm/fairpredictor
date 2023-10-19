import importlib.util
import json
import os
import shutil
import time
import uuid

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
    use_raster2polygon=False,
    remove_metadata=True,
):
    """
    Parameters:
        bbox : Bounding box of the area you want to run prediction on
        model_path : Path of your downloaded model checkpoint
        zoom_level : Zoom level of the tiles to be used for prediction
        tms_url : Your Image URL on which you want to detect feature
        tile_size : Optional >> Tile size to be used in pixel default : 256*256
    """
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
        )
    print(f"It took {round(time.time()-start)} sec to extract polygons")
    with open(geojson_path, "r") as f:
        prediction_geojson_data = json.load(f)
    if remove_metadata:
        shutil.rmtree(base_path)
    return prediction_geojson_data
