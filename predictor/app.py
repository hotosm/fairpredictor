import json
import os
import shutil
import time
import uuid

from geomltoolkits.downloader import tms as TMSDownloader
from geomltoolkits.regularizer import VectorizeMasks
from geomltoolkits.utils import merge_rasters, validate_polygon_geometries

from .prediction import run_prediction
from .utils import download_or_validate_model, morphological_cleaning


async def predict(
    model_path,
    zoom_level,
    tms_url='"https://apps.kontur.io/raster-tiler/oam/mosaic/{z}/{x}/{y}.png"',
    output_path=None,
    confidence=0.5,
    area_threshold=3,
    tolerance=0.5,
    orthogonalize=True,
    bbox=None,
    geojson=None,
    debug=False,
    get_predictions_as_points=True,
    ortho_skew_tolerance_deg=15,
    ortho_max_angle_change_deg=15,
    make_geoms_valid=True,
):
    """Detect buildings using ML model and return as GeoJSON.

    Parameters:
        model_path: Path of downloaded model checkpoint
        zoom_level: Zoom level for prediction tiles
        tms_url: Image URL for feature detection
        output_path: Directory to save prediction results (temporary UUID dir if None)
        confidence: Threshold for filtering predictions (0-1)
        area_threshold: Minimum polygon area in sqm (default: 3)
        tolerance: Simplification tolerance in meters (default: 0.5)
        orthogonalize: Whether to square building corners
        bbox: Bounding box for prediction area
        geojson: GeoJSON object for prediction area
        debug: Whether to produce merged input images and keep intermediate files
        get_predictions_as_points: Whether to generate point representations
        ortho_skew_tolerance_deg: Max skew angle for orthogonalization (0-45)
        ortho_max_angle_change_deg: Max corner adjustment angle (0-45)
    """
    if not bbox and not geojson:
        raise ValueError("Either bbox or geojson must be provided")
    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")

    base_path = output_path or os.path.join(
        os.getcwd(), "predictions", str(uuid.uuid4())
    )
    model_path = download_or_validate_model(model_path)

    os.makedirs(base_path, exist_ok=True)
    meta_path, results_path = (
        os.path.join(base_path, "meta"),
        os.path.join(base_path, "results"),
    )
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
    )

    if debug:
        try:
            merge_rasters(
                image_download_path, os.path.join(meta_path, "merged_image_chips.tif")
            )
        except Exception as e:
            print(f"Could not merge input images: {e}")

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
    prediction_merged_mask_path = os.path.join(meta_path, "merged_prediction_mask.tif")
    os.makedirs(os.path.dirname(prediction_merged_mask_path), exist_ok=True)

    merge_rasters(prediction_path, prediction_merged_mask_path)
    prediction_poly_geojson_path = os.path.join(geojson_path, "predictions.geojson")
    morphological_cleaning(prediction_merged_mask_path)
    gdf = VectorizeMasks(
        simplify_tolerance=tolerance,
        min_area=area_threshold,
        orthogonalize=orthogonalize,
        tmp_dir=os.path.join(base_path, "tmp"),
        ortho_skew_tolerance_deg=ortho_skew_tolerance_deg,
        ortho_max_angle_change_deg=ortho_max_angle_change_deg,
    ).convert(prediction_merged_mask_path, prediction_poly_geojson_path)
    print(f"It took {round(time.time() - start)} sec to extract polygons")

    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    elif not gdf.crs:
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf = gdf.to_crs("EPSG:4326")

    gdf["building"], gdf["source"] = "yes", "fAIr"

    if not debug:
        shutil.rmtree(meta_path)


    prediction_geojson_data = json.loads(gdf.to_json())
    if make_geoms_valid:
        prediction_geojson_data = validate_polygon_geometries(
            prediction_geojson_data, output_path=prediction_poly_geojson_path
        )
    if type(prediction_geojson_data) == str:
        if os.path.exists(prediction_geojson_data):
            with open(prediction_geojson_data, "r",encoding='utf-8') as f:
                prediction_geojson_data = f.read()
                prediction_geojson_data = json.loads(prediction_geojson_data)

    if get_predictions_as_points:
        gdf_points = gdf.copy()
        gdf_points.geometry = gdf_points.geometry.apply(
            lambda geom: geom.representative_point()
        )
        gdf_points.to_file(
            os.path.join(geojson_path, "predictions_points.geojson"), driver="GeoJSON"
        )
        if not output_path:
            shutil.rmtree(base_path)
        return json.loads(gdf_points.to_json())

    if not output_path:
        shutil.rmtree(base_path)

    return prediction_geojson_data
