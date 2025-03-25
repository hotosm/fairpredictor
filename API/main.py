import logging
import os
import tempfile
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

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
        "checkpoint": "https://fair-dev.hotosm.org/api/v1/workspace/download/training_324/checkpoint.tflite",
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

    orthogonalize: Optional[bool] = Field(
        True,
        description="Indicates whether to use JOSM query. Defaults to True.",
    )

    confidence: Optional[int] = Field(
        50,
        description="Threshold probability for filtering out low-confidence predictions. Defaults to 50.",
    )

    tolerance: Optional[float] = Field(
        0.5,
        description="Tolerance parameter for simplifying polygons. Defaults to 0.5.",
    )

    area_threshold: Optional[float] = Field(
        3,
        description="Threshold for filtering polygon areas. Defaults to 3.",
    )
    vectorization_algorithm: str = Field(
        "rasterio",
        description="Vectorization algorithm to adopt : potrace or rasterio",
    )

    @field_validator("tolerance")
    def validate_tolerance(cls, value):
        if value is not None:
            if value < 0 or value > 10:
                raise ValueError(f"Value should be between 0 and 10: {value}")
        return value

    @field_validator("area_threshold")
    def validate_area_threshold(cls, value):
        if value is not None:
            if value < 0 or value > 20:
                raise ValueError(f"Value should be between 0 and 20: {value}")
        return value

    @field_validator("confidence")
    def validate_confidence(cls, value):
        if value is not None:
            if value < 0 or value > 100:
                raise ValueError(f"Value should be between 0 and 100: {value}")
        return value / 100

    @field_validator("bbox")
    def validate_bbox(cls, value):
        if len(value) != 4:
            raise ValueError("bbox should have exactly 4 elements")
        return value

    @field_validator("zoom_level")
    def validate_zoom_level(cls, value):
        if value < 18 or value > 22:
            raise ValueError("Zoom level should be between 18 and 22")
        return value

    @field_validator("checkpoint")
    def validate_checkpoint(cls, value):
        """
        Validates checkpoint parameter. If URL, download the file to temp directory.
        """
        if value.startswith("http"):
            try:
                response = requests.get(value)
                response.raise_for_status()
                
                # Get the file extension from the URL or default to .tflite
                file_ext = os.path.splitext(value)[-1]
                if not file_ext:
                    file_ext = ".tflite"
                    
                _, temp_file_path = tempfile.mkstemp(suffix=file_ext)
                with open(temp_file_path, "wb") as f:
                    f.write(response.content)
                return temp_file_path
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to download model checkpoint: {e}")
        elif not os.path.exists(value):
            raise ValueError("Model checkpoint file not found")
        return value


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


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
        predictions = await predict(
            bbox=request.bbox,
            model_path=request.checkpoint,
            zoom_level=request.zoom_level,
            tms_url=request.source,
            confidence=request.confidence,
            tolerance=request.tolerance,
            area_threshold=request.area_threshold,
            orthogonalize=request.orthogonalize,
            vectorization_algorithm=request.vectorization_algorithm,
        )
        
        if request.checkpoint.startswith("/tmp") and os.path.exists(request.checkpoint):
            try:
                os.remove(request.checkpoint)
            except Exception:
                pass
                
        return predictions
    except Exception as e:
        # raise e
        logging.error(f"Failed to run prediction: {e}")
        raise HTTPException(status_code=500, detail=str('Failed to run prediction: '))