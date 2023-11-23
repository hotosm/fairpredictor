import os
import tempfile
from typing import List, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field, PositiveFloat, validator

from predictor import predict

app = FastAPI(
    title="fAIr Prediction API",
    description="Standalone API for Running .h5, .tf, .tflite Model Predictions",
)


class PredictionRequest(BaseModel):
    """
    Request model for the prediction endpoint.
    """

    bbox: List[float]

    checkpoint: str = Field(
        ...,
        example="path/to/model.tflite or https://example.com/model.tflite",
        description="Path or URL to the machine learning model file.",
    )

    zoom_level: int = Field(
        ...,
        description="Zoom level of the tiles to be used for prediction.",
    )

    source: str = Field(
        ...,
        description="Your Image URL on which you want to detect features.",
    )

    use_josm_q: Optional[bool] = Field(
        False,
        description="Indicates whether to use JOSM query. Defaults to False.",
    )

    merge_adjacent_polygons: Optional[bool] = Field(
        True,
        description="Merges adjacent self-intersecting or containing each other polygons. Defaults to True.",
    )

    confidence: Optional[int] = Field(
        50,
        description="Threshold probability for filtering out low-confidence predictions. Defaults to 50.",
    )

    max_angle_change: Optional[int] = Field(
        15,
        description="Maximum angle change parameter for prediction. Defaults to 15.",
    )

    skew_tolerance: Optional[int] = Field(
        15,
        description="Skew tolerance parameter for prediction. Defaults to 15.",
    )

    tolerance: Optional[float] = Field(
        0.5,
        description="Tolerance parameter for simplifying polygons. Defaults to 0.5.",
    )

    area_threshold: Optional[float] = Field(
        3,
        description="Threshold for filtering polygon areas. Defaults to 3.",
    )

    tile_overlap_distance: Optional[float] = Field(
        0.15,
        description="Provides tile overlap distance to remove the strip between predictions. Defaults to 0.15.",
    )

    @validator(
        "max_angle_change",
        "skew_tolerance",
        "tolerance",
        "tile_overlap_distance",
        "area_threshold",
    )
    def validate_values(self, value):
        if value is not None:
            if value < 0 or value > 45:
                raise ValueError(f"Value should be between 0 and 45: {value}")
        return value

    @validator("tolerance")
    def validate_tolerance(self, value):
        if value is not None:
            if value < 0 or value > 10:
                raise ValueError(f"Value should be between 0 and 10: {value}")
        return value

    @validator("tile_overlap_distance")
    def validate_tile_overlap_distance(self, value):
        if value is not None:
            if value < 0 or value > 1:
                raise ValueError(f"Value should be between 0 and 1: {value}")
        return value

    @validator("area_threshold")
    def validate_area_threshold(self, value):
        if value is not None:
            if value < 0 or value > 20:
                raise ValueError(f"Value should be between 0 and 20: {value}")
        return value

    @validator("confidence")
    def validate_confidence(self, value):
        if value is not None:
            if value < 0 or value > 100:
                raise ValueError(f"Value should be between 0 and 100: {value}")
        return value / 100

    @validator("bbox")
    def validate_bbox(self, value):
        if len(value) != 4:
            raise ValueError("bbox should have exactly 4 elements")
        return value

    @validator("zoom_level")
    def validate_zoom_level(self, value):
        if value < 18 or value > 22:
            raise ValueError("Zoom level should be between 18 and 22")
        return value


@app.post("/predict/")
async def predict_api(request: PredictionRequest):
    """
    Endpoint to predict results based on specified parameters.

    Parameters:
    - `request` (PredictionRequest): Request body containing prediction parameters.

    Returns:
    - Predicted results.
    """
    try:
        predictions = predict(
            bbox=request.bbox,
            model_path=request.checkpoint,
            zoom_level=request.zoom_level,
            tms_url=request.source,
            tile_size=256,
            confidence=request.confidence,
            tile_overlap_distance=request.tile_overlap_distance,
            merge_adjancent_polygons=request.merge_adjacent_polygons,
            max_angle_change=request.max_angle_change,
            skew_tolerance=request.skew_tolerance,
            tolerance=request.tolerance,
            area_threshold=request.area_threshold,
        )
        return predictions
    except Exception as e:
        return {"error": str(e)}
