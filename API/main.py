import importlib.metadata
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from predictor import predict
from predictor.models import PredictionRequest

load_dotenv()

__version__ = importlib.metadata.version("fairpredictor")

logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="fAIr Prediction API",
    description="Geospatial Machine Learning Prediction Service",
    version=__version__,
    contact={"name": "fAIr Support", "email": "tech@hotosm.org"},
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
)

app.state.limiter = limiter

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
        response.headers["X-Process-Time"] = f"{time.time() - start_time:.4f} seconds"
        return response


app.add_middleware(TimingMiddleware)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "fAIr Prediction API",
        "version": __version__,
        "description": "Geospatial Machine Learning Prediction Service",
        "documentation": {"swagger_ui": "/docs", "redoc": "/redoc", "openapi_json": "/openapi.json"},
        "endpoints": {"health": "/health", "predict": "/predict"},
        "contact": "tech@hotosm.org",
        "license": "MIT",
    }


@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    health_status = {"status": "healthy", "services": {}}

    try:
        import geomltoolkits  # noqa: F401

        health_status["services"]["geomltoolkits"] = {
            "available": True,
            "version": importlib.metadata.version("geomltoolkits"),
        }
    except ImportError:
        health_status["services"]["geomltoolkits"] = {"available": False, "version": None}

    return health_status


@app.post("/predict/")
@limiter.limit(os.getenv("PREDICT_RATE_LIMIT", "5/minute"))
async def predict_api(params: PredictionRequest, request: Request):
    try:
        predictions_result = await predict(
            bbox=params.bbox,
            model_path=params.checkpoint,
            zoom_level=params.zoom_level,
            tms_url=params.source,
            confidence=params.confidence / 100,
            tolerance=params.tolerance,
            area_threshold=params.area_threshold,
            orthogonalize=params.orthogonalize,
            ortho_skew_tolerance_deg=params.ortho_skew_tolerance_deg,
            ortho_max_angle_change_deg=params.ortho_max_angle_change_deg,
            get_predictions_as_points=params.get_predictions_as_points,
            make_geoms_valid=params.make_geoms_valid,
            task=params.task,
        )

        if params.checkpoint.startswith("/tmp") and os.path.exists(params.checkpoint):
            try:
                os.remove(params.checkpoint)
            except Exception as cleanup_error:
                logger.warning("Failed to cleanup temp file: %s", cleanup_error)

        return predictions_result
    except RuntimeError as e:
        error_message = str(e)
        logger.warning("Runtime error during prediction: %s", error_message)

        if "No images found" in error_message:
            raise HTTPException(
                status_code=404,
                detail="No images found in the specified area. Please check your bbox and source URL.",
            ) from e
        raise HTTPException(status_code=400, detail=f"Prediction failed: {error_message}") from e
    except Exception as e:
        logger.error("Prediction failed with unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}") from e


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Incoming request: %s %s", request.method, request.url)
    response = await call_next(request)
    logger.info("Response status: %s", response.status_code)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
    )
