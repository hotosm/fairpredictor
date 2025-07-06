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
    ortho_skew_tolerance_deg=15,
    ortho_max_angle_change_deg=15,
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
        remove_metadata: Whether to delete intermediate files after processing
        orthogonalize: Whether to square building corners
        bbox: Bounding box for prediction area
        geojson: GeoJSON object for prediction area
        merge_input_images_to_single_image: Whether to merge source images
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
    prediction_merged_mask_path = os.path.join(meta_path, "merged_prediction_mask.tif")
    os.makedirs(os.path.dirname(prediction_merged_mask_path), exist_ok=True)

    merge_rasters(prediction_path, prediction_merged_mask_path)
    gdf = VectorizeMasks(
        simplify_tolerance=tolerance,
        min_area=area_threshold,
        orthogonalize=orthogonalize,
        tmp_dir=os.path.join(base_path, "tmp"),
        ortho_skew_tolerance_deg=ortho_skew_tolerance_deg,
        ortho_max_angle_change_deg=ortho_max_angle_change_deg,
    ).convert(
        prediction_merged_mask_path, os.path.join(geojson_path, "predictions.geojson")
    )
    print(f"It took {round(time.time() - start)} sec to extract polygons")

    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    elif not gdf.crs:
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf = gdf.to_crs("EPSG:4326")

    gdf["building"], gdf["source"] = "yes", "fAIr"

    if remove_metadata:
        shutil.rmtree(meta_path)

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

    prediction_geojson_data = json.loads(gdf.to_json())

    if not output_path:
        shutil.rmtree(base_path)

    return prediction_geojson_data
