from .downloader import download
from .prediction import run_prediction
from .vectorizer import vectorize 
import json
import shutil

def predict(bbox,model_path,zoom_level,tms_url,tile_size=256):
    image_download_path = download(bbox,zoom_level=zoom_level,tms_url=tms_url,tile_size=tile_size)
    prediction_path = run_prediction(model_path,image_download_path)
    geojson_path = vectorize(prediction_path)
    with open(geojson_path, "r") as f:
        prediction_geojson_data = json.load(f)
    shutil.rmtree('/tmp')
    return prediction_geojson_data