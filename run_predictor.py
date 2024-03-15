"""
Runs prediction for fAIr modules 

Author : 
Kshitij raj sharma
Kshitij.sharma@hotosm.org

Example : 
python run_predictor.py --model_id 121 --zoom_level 20 --tms_url "https://tiles.openaerialmap.org/65e4bb85e6f8d4000128235a/0/65e4bb85e6f8d4000128235b/{z}/{x}/{y}" --bbox -10.7933 6.3737 -10.7921 6.3749

"""

import argparse
import shutil

import geopandas as gpd
import matplotlib.pyplot as plt
import requests

from predictor import predict

BASE_FAIR_API_URL = "https://fair-dev.hotosm.org/api/v1"


def get_model(model_id):
    model_meta = requests.get(f"{BASE_FAIR_API_URL}/model/{model_id}/")
    model_meta.raise_for_status()
    model_meta = model_meta.json()
    print(model_meta)
    model_path = "checkpoint.h5"
    url = f'{BASE_FAIR_API_URL}/workspace/download/dataset_{model_meta["dataset"]}/output/training_{model_meta["published_training"]}/checkpoint.h5'

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(model_path, "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    return model_path


def display_predictions_gdf(predictions):
    predictions_gdf = gpd.GeoDataFrame.from_features(predictions)
    predictions_gdf.plot()
    plt.show()


def main(model_id, zoom_level, tms_url, bbox):
    model_path = get_model(model_id)
    predictions = predict(
        bbox,
        model_path,
        zoom_level,
        tms_url,
        tile_overlap_distance=0.01,
        remove_metadata=False,
    )
    print(predictions)
    display_predictions_gdf(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict objects from satellite imagery."
    )
    parser.add_argument(
        "--model_id", type=int, help="Model ID Published on fAIr", required=True
    )
    parser.add_argument(
        "--zoom_level",
        type=int,
        help="Zoom level of the tiles to be used for prediction",
        required=True,
    )
    parser.add_argument(
        "--tms_url",
        type=str,
        help="Your Image URL on which you want to detect feature",
        required=True,
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        help="Bounding box for prediction [min_lon, min_lat, max_lon, max_lat]",
        required=True,
    )

    args = parser.parse_args()

    main(args.model_id, args.zoom_level, args.tms_url, args.bbox)
