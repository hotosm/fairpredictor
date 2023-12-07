import os
import tempfile
from typing import List, Optional

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, PositiveFloat, validator

from predictor import predict

app = FastAPI(
    title="fAIr Prediction API",
    description="Standalone API for Running .h5, .tf, .tflite Model Predictions",
)


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """
    Request model for the prediction endpoint.

    Example :
    {
        "bbox": [
            100.56228021333352,
            13.685230854641182,
            100.56383321235313,
            13.685961853747969
        ],
        "checkpoint": "https://fair-dev.hotosm.org/api/v1/workspace/download/dataset_58/output/training_324//checkpoint.tflite",
        "zoom_level": 20,
        "source": "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"
    }
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
    )
    def validate_values(cls, value):
        if value is not None:
            if value < 0 or value > 45:
                raise ValueError(f"Value should be between 0 and 45: {value}")
        return value

    @validator("tolerance")
    def validate_tolerance(cls, value):
        if value is not None:
            if value < 0 or value > 10:
                raise ValueError(f"Value should be between 0 and 10: {value}")
        return value

    @validator("tile_overlap_distance")
    def validate_tile_overlap_distance(cls, value):
        if value is not None:
            if value < 0 or value > 1:
                raise ValueError(f"Value should be between 0 and 1: {value}")
        return value

    @validator("area_threshold")
    def validate_area_threshold(cls, value):
        if value is not None:
            if value < 0 or value > 20:
                raise ValueError(f"Value should be between 0 and 20: {value}")
        return value

    @validator("confidence")
    def validate_confidence(cls, value):
        if value is not None:
            if value < 0 or value > 100:
                raise ValueError(f"Value should be between 0 and 100: {value}")
        return value / 100

    @validator("bbox")
    def validate_bbox(cls, value):
        if len(value) != 4:
            raise ValueError("bbox should have exactly 4 elements")
        return value

    @validator("zoom_level")
    def validate_zoom_level(cls, value):
        if value < 18 or value > 22:
            raise ValueError("Zoom level should be between 18 and 22")
        return value

    @validator("checkpoint")
    def validate_checkpoint(cls, value):
        """
        Validates checkpoint parameter. If URL, download the file to temp directory.
        """
        if value.startswith("http"):
            response = requests.get(value)
            if response.status_code != 200:
                raise ValueError(
                    "Failed to download model checkpoint from the provided URL"
                )
            _, temp_file_path = tempfile.mkstemp(suffix=".tflite")
            with open(temp_file_path, "wb") as f:
                f.write(response.content)
            return temp_file_path
        elif not os.path.exists(value):
            raise ValueError("Model checkpoint file not found")
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
