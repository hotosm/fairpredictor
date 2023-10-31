import os
import tempfile

import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field, FilePath, HttpUrl, validator

from predictor import predict

app = FastAPI(
    title="fAIr Prediction API",
    description="Standalone API for Running .h5,.tf,.tflite Model Predictions",
)


class PredictionRequest(BaseModel):
    """
    Represents the request body for making predictions.

    Parameters:
    - `bbox` (list[float]): Bounding box coordinates [min_longitude, min_latitude, max_longitude, max_latitude].
    - `checkpoint` (str): Path or URL to the machine learning model file.
    - `zoom` (int): Zoom level for predictions (between 18 and 23).
    - `tms` (str): URL for tile map service.

    Example:
    ```
    {
        "bbox": [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969],
        "checkpoint": "path/to/model.tflite",
        "zoom": 20,
        "tms": "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"
    }
    """

    bbox: list[float] = Field(
        ...,
        example=[
            100.56228021333352,
            13.685230854641182,
            100.56383321235313,
            13.685961853747969,
        ],
    )
    checkpoint: str = Field(
        ..., example="path/to/model.tflite or https://example.com/model.tflite"
    )
    zoom: int = Field(..., ge=18, le=23, example=20)
    tms: str = Field(
        ...,
        example="https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}",
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
            request.bbox, request.checkpoint, request.zoom, request.tms
        )
        return predictions
    except Exception as e:
        return {"error": str(e)}
