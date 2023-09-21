import json
import shutil

from .downloader import download
from .prediction import run_prediction
from .vectorizer import vectorize


def predict(bbox, model_path, zoom_level, tms_url, tile_size=256):
    """
    Parameters:
        bbox : Bounding box of the area you want to run prediction on
        model_path : Path of your downloaded model checkpoint
        zoom_level : Zoom level of the tiles to be used for prediction
        tms_url : Your Image URL on which you want to detect feature
        tile_size : Optional >> Tile size to be used in pixel default : 256*256
    """
    image_download_path = download(
        bbox, zoom_level=zoom_level, tms_url=tms_url, tile_size=tile_size
    )
    prediction_path = run_prediction(model_path, image_download_path)
    geojson_path = vectorize(prediction_path)
    with open(geojson_path, "r") as f:
        prediction_geojson_data = json.load(f)
    # shutil.rmtree('/tmp')
    return prediction_geojson_data
