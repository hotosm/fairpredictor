import importlib.metadata
import logging
import os
import subprocess
import tempfile
import time
from typing import List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from predictor import predict

load_dotenv()


__version__ = importlib.metadata.version("fairpredictor")

logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="fAIr Prediction API",
    description="""
    Geospatial Machine Learning Prediction Service

    This API provides advanced feature detection and vectorization capabilities 
    for geospatial data processing. It supports various machine learning model 
    predictions with configurable parameters.

    Key Features
    - Flexible machine learning model predictions
    - Customizable confidence and tolerance thresholds
    - Multiple vectorization algorithms
    - Comprehensive error handling and validation
    """,
    version=__version__,
    contact={
        "name": "fAIr Support",
        "email": "tech@hotosm.org",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add limiter to app state
app.state.limiter = limiter

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "*").split(",")
allowed_methods = os.getenv("CORS_ALLOW_METHODS", "GET,POST,OPTIONS").split(",")
allowed_headers = os.getenv("CORS_ALLOW_HEADERS", "*").split(",")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=allowed_methods,
    allow_headers=allowed_headers,
)


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f} seconds"
        return response


app.add_middleware(TimingMiddleware)


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

    orthogonalize: Optional[bool] = Field(
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
        default=os.getenv("DEFAULT_VECOTRIZATION_ALGORITHM", "rasterio"),
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


@app.get("/", include_in_schema=False)
async def root():
    """
    Root endpoint that returns API information and documentation links.
    """
    return {
        "name": "fAIr Prediction API",
        "version": __version__,
        "description": "Geospatial Machine Learning Prediction Service",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {"health": "/health", "predict": "/predict"},
        "contact": "tech@hotosm.org",
        "license": "MIT",
    }


@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    health_status = {"status": "healthy", "services": {}}

    try:
        potrace_version = subprocess.check_output(
            ["potrace", "--version"], stderr=subprocess.STDOUT, text=True
        )
        health_status["services"]["potrace"] = {
            "available": True,
            "version": potrace_version.strip().split("\n")[0],
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        health_status["services"]["potrace"] = {"available": False, "version": None}

    return health_status


@app.post("/predict/")
@limiter.limit(os.getenv("PREDICT_RATE_LIMIT", "5/minute"))
async def predict_api(params: PredictionRequest, request: Request):
    try:
        predictions = await predict(
            bbox=params.bbox,
            model_path=params.checkpoint,
            zoom_level=params.zoom_level,
            tms_url=params.source,
            confidence=params.confidence / 100,  # Convert percentage to decimal
            tolerance=params.tolerance,
            area_threshold=params.area_threshold,
            orthogonalize=params.orthogonalize,
            vectorization_algorithm=params.vectorization_algorithm,
        )

        # Clean up temporary files
        if params.checkpoint.startswith("/tmp") and os.path.exists(params.checkpoint):
            try:
                os.remove(params.checkpoint)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

        return predictions
    except RuntimeError as e:
        error_message = str(e)
        logger.warning(f"Runtime error during prediction: {error_message}")

        if "No images found" in error_message:
            raise HTTPException(
                status_code=404,
                detail="No images found in the specified area. Please check your bbox and source URL.",
            )
        else:
            # Other runtime errors - could be client or server issue
            raise HTTPException(
                status_code=400, detail=f"Prediction failed: {error_message}"
            )
    except Exception as e:
        # Unexpected errors - likely server issues
        logger.error(f"Prediction failed with unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error during prediction: {str(e)}"
        )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
    )
