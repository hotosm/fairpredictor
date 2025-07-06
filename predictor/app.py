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
    model_path,
    zoom_level,
    tms_url='"https://apps.kontur.io/raster-tiler/oam/mosaic/{z}/{x}/{y}.png"',
    output_path=None,
    confidence=0.5,
    area_threshold=3,
    tolerance=0.5,
    remove_metadata=True,
    orthogonalize=True,
    bbox=None,
    geojson=None,
    merge_input_images_to_single_image=False,
    get_predictions_as_points=True,
    ortho_skew_tolerance_deg: int = 15,  # angle (0,45> degrees
    ortho_max_angle_change_deg: int = 15,  # angle (0,45> degrees
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
    if not bbox and not geojson:
        raise ValueError("Either bbox or geojson must be provided")
    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")
    if output_path:
        base_path = output_path
    else:
        base_path = os.path.join(os.getcwd(), "predictions", str(uuid.uuid4()))

    model_path = download_or_validate_model(model_path)
    os.makedirs(base_path, exist_ok=True)
    meta_path = os.path.join(output_path, "meta")
    results_path = os.path.join(output_path, "results")
    os.makedirs(meta_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    image_download_path = os.path.join(meta_path, "image")
    os.makedirs(image_download_path, exist_ok=True)

    image_download_path = await TMSDownloader.download_tiles(
        bbox=bbox,
        geojson=geojson,
        zoom=zoom_level,
        tms=tms_url,
        out=image_download_path,
        georeference=True,
        crs="3857",
        # dump=True,
    )
    if merge_input_images_to_single_image:
        merge_rasters(
            image_download_path, os.path.join(meta_path, "merged_image_chips.tif")
        )

    prediction_path = os.path.join(meta_path, "prediction")
    os.makedirs(prediction_path, exist_ok=True)

    prediction_path = run_prediction(
        model_path,
        image_download_path,
        prediction_path=prediction_path,
        confidence=confidence,
        crs="3857",
    )
    start = time.time()

    geojson_path = os.path.join(results_path, "geojson")
    os.makedirs(geojson_path, exist_ok=True)
    prediction_geojson_path = os.path.join(geojson_path, "predictions.geojson")

    prediction_merged_mask_path = os.path.join(meta_path, "merged_prediction_mask.tif")
    os.makedirs(os.path.dirname(prediction_merged_mask_path), exist_ok=True)

    # Merge rasters
    merge_rasters(prediction_path, prediction_merged_mask_path)
    tmp_dir = os.path.join(base_path, "tmp")
    converter = VectorizeMasks(
        simplify_tolerance=tolerance,
        min_area=area_threshold,
        orthogonalize=orthogonalize,
        tmp_dir=tmp_dir,
        ortho_skew_tolerance_deg=ortho_skew_tolerance_deg,
        ortho_max_angle_change_deg=ortho_max_angle_change_deg,
    )
    gdf = converter.convert(prediction_merged_mask_path, prediction_geojson_path)
    shutil.rmtree(tmp_dir)
    print(f"It took {round(time.time() - start)} sec to extract polygons")

    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    elif not gdf.crs:
        # if not defined assume its 3857 because above 3857 is hardcoded
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf = gdf.to_crs("EPSG:4326")

    gdf["building"] = "yes"
    gdf["source"] = "fAIr"

    if get_predictions_as_points:
        gdf_representative_points = gdf.copy()
        gdf_representative_points.geometry = gdf_representative_points.geometry.apply(
            lambda geom: geom.representative_point()
        )
        gdf_representative_points.to_file(
            os.path.join(geojson_path, "prediction_points.geojson"), driver="GeoJSON"
        )
    prediction_geojson_data = json.loads(gdf.to_json())

    if remove_metadata:
        shutil.rmtree(meta_path)
    if not output_path:
        shutil.rmtree(base_path)
    return prediction_geojson_data
