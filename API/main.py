import os
import tempfile

import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field, PositiveFloat, validator

from predictor import predict

app = FastAPI(
    title="fAIr Prediction API",
    description="Standalone API for Running .h5, .tf, .tflite Model Predictions",
)


class PredictionRequest(BaseModel):
    bbox: list[float] = Field(
        ...,
        example=[
            100.56228021333352,
            13.685230854641182,
            100.56383321235313,
            13.685961853747969,
        ],
        description="Bounding box coordinates [min_longitude, min_latitude, max_longitude, max_latitude].",
    )
    checkpoint: str = Field(
        ...,
        example="path/to/model.tflite or https://example.com/model.tflite",
        description="Path or URL to the machine learning model file.",
    )
    zoom: int = Field(
        ...,
        ge=18,
        le=23,
        example=20,
        description="Zoom level for predictions (between 18 and 23).",
    )
    tms: str = Field(
        ...,
        example="https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}",
        description="URL for tile map service.",
    )
    tile_size: int = Field(
        256,
        example=256,
        description="Tile size in pixels. Defaults to 256*256.",
    )
    base_path: str = Field(
        None,
        example="/path/to/working/directory",
        description="Base path for working directory. Defaults to None.",
    )
    confidence: float = Field(
        0.5,
        example=0.5,
        gt=0,
        le=1,
        description="Threshold probability for filtering out low-confidence predictions. Defaults to 0.5.",
    )
    area_threshold: PositiveFloat = Field(
        3,
        example=3,
        description="Threshold for filtering polygon areas. Defaults to 3 sqm.",
    )
    tolerance: PositiveFloat = Field(
        0.5,
        example=0.5,
        description="Tolerance parameter for simplifying polygons. Defaults to 0.5 m.",
    )
    tile_overlap_distance: PositiveFloat = Field(
        0.15,
        example=0.15,
        description="Tile overlap distance to remove the strip between predictions. Defaults to 0.15 m.",
    )
    merge_adjacent_polygons: bool = Field(
        True,
        example=True,
        description="Flag to merge adjacent polygons. Defaults to True.",
    )

    @validator("bbox")
    def validate_bbox_length(cls, value):
        """
        Validates the length of bbox coordinates.
        """
        if len(value) != 4:
            raise ValueError("bbox must contain 4 float values")
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
            request.bbox,
            request.checkpoint,
            request.zoom,
            request.tms,
            confidence=request.confidence,
            area_threshold=request.area_threshold,
            tolerance=request.tolerance,
            tile_overlap_distance=request.tile_overlap_distance,
            merge_adjancent_polygons=request.merge_adjacent_polygons,
        )
        return predictions
    except Exception as e:
        return {"error": str(e)}
