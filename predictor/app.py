import json
import logging
import os
import shutil
import time
import uuid
from typing import Literal

import numpy as np
import rasterio
import rasterio.features
from geomltoolkits import merge_rasters, morphological_cleaning, validate_polygon_geometries, vectorize_mask
from geomltoolkits.downloader import tms as TMSDownloader
from shapely.geometry import mapping

from .prediction import run_prediction
from .utils import download_or_validate_model, threshold_mask

logger = logging.getLogger(__name__)


def _compute_polygon_confidence(
    gdf,
    raw_raster_path: str,
) -> list[float]:
    """Compute mean confidence per polygon from the raw prediction raster."""
    with rasterio.open(raw_raster_path) as src:
        raw_data = src.read(1).astype(np.float32) / 255.0
        transform = src.transform
        raster_crs = src.crs

    gdf_projected = gdf.to_crs(raster_crs) if gdf.crs and str(gdf.crs) != str(raster_crs) else gdf

    confidences = []
    for geom in gdf_projected.geometry:
        mask = rasterio.features.geometry_mask(
            [mapping(geom)],
            out_shape=raw_data.shape,
            transform=transform,
            invert=True,
        )
        pixels = raw_data[mask]
        if len(pixels) > 0:
            confidences.append(round(float(np.mean(pixels)), 4))
        else:
            confidences.append(0.0)

    return confidences


def _threshold_raster(input_path: str, output_path: str, confidence: float) -> None:
    """Read a raw confidence raster, threshold to binary, and write to output."""
    with rasterio.open(input_path) as src:
        raw = src.read(1)
        profile = src.profile.copy()

    threshold_value = int(confidence * 255)
    binary = threshold_mask(raw, threshold=threshold_value)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(binary, 1)


async def predict(
    model_path: str,
    zoom_level: int,
    tms_url: str = "https://apps.kontur.io/raster-tiler/oam/mosaic/{z}/{x}/{y}.png",
    output_path: str | None = None,
    confidence: float = 0.5,
    area_threshold: float = 3,
    tolerance: float = 0.5,
    orthogonalize: bool = True,
    bbox: list[float] | None = None,
    geojson: dict | str | None = None,
    debug: bool = False,
    get_predictions_as_points: bool = True,
    ortho_skew_tolerance_deg: int = 15,
    ortho_max_angle_change_deg: int = 15,
    make_geoms_valid: bool = True,
    task: Literal["segmentation", "detection", "classification"] = "segmentation",
) -> dict:
    if task != "segmentation":
        raise NotImplementedError(f"Task '{task}' is not yet supported. Only 'segmentation' is available.")

    if not bbox and not geojson:
        raise ValueError("Either bbox or geojson must be provided")
    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")

    base_path = output_path or os.path.join(os.getcwd(), "predictions", str(uuid.uuid4()))
    model_path = download_or_validate_model(model_path)

    os.makedirs(base_path, exist_ok=True)
    meta_path = os.path.join(base_path, "meta")
    results_path = os.path.join(base_path, "results")
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
            merge_rasters(image_download_path, os.path.join(meta_path, "merged_image_chips.tif"))
        except Exception as e:
            logger.warning("Could not merge input images: %s", e)

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

    raw_merged_path = os.path.join(meta_path, "merged_raw_confidence.tif")
    binary_merged_path = os.path.join(meta_path, "merged_prediction_mask.tif")

    merge_rasters(prediction_path, raw_merged_path)
    _threshold_raster(raw_merged_path, binary_merged_path, confidence)
    morphological_cleaning(binary_merged_path)

    prediction_poly_geojson_path = os.path.join(geojson_path, "predictions.geojson")
    gdf = vectorize_mask(
        input_tiff=binary_merged_path,
        output_geojson=prediction_poly_geojson_path,
        simplify_tolerance=tolerance,
        min_area=area_threshold,
        orthogonalize=orthogonalize,
        ortho_skew_tolerance_deg=ortho_skew_tolerance_deg,
        ortho_max_angle_change_deg=ortho_max_angle_change_deg,
    )

    if len(gdf) > 0:
        gdf["confidence"] = _compute_polygon_confidence(gdf, raw_merged_path)

    logger.info("Polygon extraction took %d sec", round(time.time() - start))

    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    elif not gdf.crs:
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf = gdf.to_crs("EPSG:4326")

    gdf["building"], gdf["source"] = "yes", "fAIr"

    if not debug:
        shutil.rmtree(meta_path)

    prediction_geojson_data = json.loads(gdf.to_json())
    if make_geoms_valid and len(gdf) > 0:
        prediction_geojson_data = validate_polygon_geometries(
            prediction_geojson_data, output_path=prediction_poly_geojson_path
        )
    if isinstance(prediction_geojson_data, str) and os.path.exists(prediction_geojson_data):
        with open(prediction_geojson_data, encoding="utf-8") as f:
            prediction_geojson_data = json.loads(f.read())

    if get_predictions_as_points:
        gdf_points = gdf.copy()
        gdf_points.geometry = gdf_points.geometry.apply(lambda geom: geom.representative_point())
        gdf_points.to_file(
            os.path.join(geojson_path, "predictions_points.geojson"),
            driver="GeoJSON",
        )
        if not output_path:
            shutil.rmtree(base_path)
        return json.loads(gdf_points.to_json())

    if not output_path:
        shutil.rmtree(base_path)

    return prediction_geojson_data
