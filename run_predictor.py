"""
Runs prediction for fAIr modules 

Author : 
Kshitij raj sharma
Kshitij.sharma@hotosm.org

Example : 
python run_predictor.py --model_id 121 --zoom_level 20 --tms_url "https://tiles.openaerialmap.org/65e4bb85e6f8d4000128235a/0/65e4bb85e6f8d4000128235b/{z}/{x}/{y}" --bbox -10.7933 6.3737 -10.7921 6.3749 --multimask

"""

import argparse
import json
import os
import shutil

import efficientnet.keras as efn
import geopandas as gpd
import matplotlib.pyplot as plt
import requests

from predictor import predict

BASE_FAIR_API_URL = "https://fair-dev.hotosm.org/api/v1"


def get_model(model_id):
    model_path = "checkpoint.h5"
    if os.path.exists(model_path):
        print("Model already exists, Skipping download", model_path)
        return model_path
    model_meta = requests.get(f"{BASE_FAIR_API_URL}/model/{model_id}/")
    model_meta.raise_for_status()
    model_meta = model_meta.json()
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


def main(
    model_id,
    zoom_level,
    tms_url,
    bbox,
    multimask=False,
    display=True,
    out="predictions.geojson",
    verbose=True,
    meta=True,
    mergepolys=True,
):
    model_path = get_model(model_id)
    predictions = predict(
        bbox,
        model_path,
        zoom_level,
        tms_url,
        tile_overlap_distance=0.01,
        remove_metadata=meta is False,
        multi_masks=multimask,
        verbose=verbose,
        merge_adjancent_polygons=mergepolys,
    )
    with open(out, "w") as file:
        json.dump(predictions, file, indent=4)

    if verbose:
        print("Predictions written to disk", out)
    if display:
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
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="Enable multimask prediction",
    )

    parser.add_argument(
        "--out",
        type=str,
        help="Output name for predictions eg : results.geojson",
        default="predictions.geojson",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display predictions using matplotlib",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display additonal log messages",
    )
    parser.add_argument(
        "--meta",
        action="store_true",
        help="Enable storing metadata during prediction",
    )

    parser.add_argument(
        "--mergepolys",
        action="store_true",
        help="Merges adjancent polygons from the prediction",
    )

    args = parser.parse_args()

    main(
        args.model_id,
        args.zoom_level,
        args.tms_url,
        args.bbox,
        args.multimask,
        args.display,
        args.out,
        args.verbose,
        args.meta,
        args.mergepolys,
    )
