import os
import tempfile
from typing import List, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Prediction Request Model for Geospatial Machine Learning
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bbox": [100.56228, 13.685230, 100.56383, 13.685961],
                "checkpoint": "https://api-prod.fair.hotosm.org/api/v1/workspace/download/ramp/baseline.tflite",
                "zoom_level": 20,
                "source": "https://apps.kontur.io/raster-tiler/oam/mosaic/{z}/{x}/{y}.png",
                "confidence": 50,
                "tolerance": 0.5,
            }
        }
    )

    bbox: List[float] = Field(
        ...,
        description="Geographical bounding box coordinates [min_lon, min_lat, max_lon, max_lat]",
        min_items=4,
        max_items=4,
    )

    checkpoint: str = Field(
        ...,
        description="URL or local path to the machine learning model file",
    )

    zoom_level: int = Field(
        ...,
        description="Zoom level for tile resolution (18-22)",
        ge=18,
        le=22,
    )

    source: str = Field(
        ...,
        description="Tile map service (TMS) URL template",
    )

    use_josm_q: Optional[bool] = Field(
        default=True,
        description="Apply orthogonalization to detected features",
    )

    confidence: Optional[int] = Field(
        default=50,
        description="Confidence threshold (0-100)",
        ge=0,
        le=100,
    )

    tolerance: Optional[float] = Field(
        default=0.5,
        description="Polygon simplification tolerance",
        ge=0,
        le=10,
    )

    area_threshold: Optional[float] = Field(
        default=3,
        description="Minimum polygon area threshold",
        ge=0,
        le=20,
    )

    vectorization_algorithm: str = Field(
        default=os.getenv("DEFAULT_VECTORIZATION_ALGORITHM", "rasterio"),
        description="Algorithm for vectorization: 'rasterio' or 'potrace'",
        pattern="^(potrace|rasterio)$",
    )

    @field_validator("checkpoint")
    def validate_checkpoint(cls, value):
        if value.startswith("http"):
            try:
                response = requests.get(value)
                response.raise_for_status()

                file_ext = os.path.splitext(value)[-1] or ".tflite"
                _, temp_file_path = tempfile.mkstemp(suffix=file_ext)

                with open(temp_file_path, "wb") as f:
                    f.write(response.content)

                return temp_file_path
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to download model checkpoint: {e}")
        elif not os.path.exists(value):
            raise ValueError("Model checkpoint file not found")
        return value
