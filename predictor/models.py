from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .utils import download_or_validate_model


class PredictionRequest(BaseModel):
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

    bbox: list[float] = Field(
        ...,
        description="Geographical bounding box coordinates [min_lon, min_lat, max_lon, max_lat]",
        min_length=4,
        max_length=4,
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

    task: Literal["segmentation", "detection", "classification"] = Field(
        default="segmentation",
        description="Prediction task type",
    )

    orthogonalize: bool = Field(
        default=True,
        description="Apply orthogonalization to detected features",
    )

    ortho_skew_tolerance_deg: int = Field(
        default=15,
        description="Skew tolerance for orthogonalization (0-45 degrees)",
        ge=0,
        le=45,
    )

    ortho_max_angle_change_deg: int = Field(
        default=15,
        description="Maximum angle change for orthogonalization (0-45 degrees)",
        ge=0,
        le=45,
    )

    confidence: int = Field(
        default=50,
        description="Confidence threshold % (0-100)",
        ge=0,
        le=100,
    )

    tolerance: float = Field(
        default=0.5,
        description="Polygon simplification tolerance",
        ge=0,
        le=10,
    )

    area_threshold: float = Field(
        default=2,
        description="Minimum polygon area threshold",
        ge=0,
        le=20,
    )

    get_predictions_as_points: bool = Field(
        default=False,
        description="Whether to include predictions as points",
    )

    output_path: str | None = Field(
        default=None,
        description="Path to save the output files",
    )

    make_geoms_valid: bool = Field(
        default=True,
        description="Whether to validate and fix polygon geometries in the output GeoJSON",
    )

    @field_validator("checkpoint")
    @classmethod
    def validate_checkpoint(cls, value: str) -> str:
        return download_or_validate_model(value)
