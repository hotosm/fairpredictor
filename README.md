## fAIr Predictor

Run fAIr model predictions anywhere. Works on CPU, compatible with serverless functions.

Requires Python >= 3.10.

## Installation

```bash
# core (RAMP / TFLite models)
uv add fairpredictor
# or
pip install fairpredictor

# with YOLO / ONNX support
uv add "fairpredictor[pytorch]"
pip install "fairpredictor[pytorch]"

# with TensorFlow support
uv add "fairpredictor[tensorflow]"
pip install "fairpredictor[tensorflow]"

# everything
uv add "fairpredictor[all]"
pip install "fairpredictor[all]"
```

## Python usage

```python
import asyncio

from predictor import DEFAULT_OAM_TMS_MOSAIC, DEFAULT_RAMP_MODEL, DEFAULT_YOLO_MODEL, predict

bbox = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
zoom_level = 20

# RAMP (TFLite) prediction
result = asyncio.run(
    predict(
        model_path=DEFAULT_RAMP_MODEL,
        zoom_level=zoom_level,
        tms_url=DEFAULT_OAM_TMS_MOSAIC,
        bbox=bbox,
        confidence=0.5,
        tolerance=0.5,
        area_threshold=3,
        orthogonalize=True,
    )
)
print(result)

# YOLO (ONNX) prediction
result = asyncio.run(
    predict(
        model_path=DEFAULT_YOLO_MODEL,
        zoom_level=zoom_level,
        tms_url=DEFAULT_OAM_TMS_MOSAIC,
        bbox=bbox,
    )
)
print(result)
```

### `predict()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | required | URL or local path to model (`.tflite`, `.onnx`, `.pt`, `.h5`) |
| `zoom_level` | `int` | required | Tile zoom level (18-22) |
| `tms_url` | `str` | OAM mosaic | TMS URL template `{z}/{x}/{y}` |
| `bbox` | `list[float]` | `None` | `[min_lon, min_lat, max_lon, max_lat]` |
| `geojson` | `dict\|str` | `None` | GeoJSON polygon as alternative to bbox |
| `confidence` | `float` | `0.5` | Confidence threshold 0-1 |
| `tolerance` | `float` | `0.5` | Polygon simplification tolerance |
| `area_threshold` | `float` | `3` | Minimum polygon area |
| `orthogonalize` | `bool` | `True` | Apply orthogonalization |
| `ortho_skew_tolerance_deg` | `int` | `15` | Orthogonalization skew tolerance (0-45) |
| `ortho_max_angle_change_deg` | `int` | `15` | Maximum angle change for orthogonalization (0-45) |
| `get_predictions_as_points` | `bool` | `True` | Include centroid points in output |
| `make_geoms_valid` | `bool` | `True` | Validate and fix output polygons |
| `task` | `str` | `"segmentation"` | Task type (`segmentation` only for now) |
| `output_path` | `str` | `None` | Directory to save outputs; auto-generated if not set |
| `debug` | `bool` | `False` | Save intermediate rasters for debugging |

Either `bbox` or `geojson` must be provided.

## Development

```bash
# Install all dependency groups
just install

# Run linting + type checking + tests
just check

# Lint only
just lint

# Tests only
just test
```

## Load testing

**Always obtain permission from the server admin before load testing.**

```bash
uv run locust -f locust.py
```

Set `HOST` to the base URL of the predictor API.

## Docker

### Build

```bash
docker build -t fairpredictor .
```

### Run

```bash
docker run --rm -p 8000:8000 fairpredictor
```

### API

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example request

```json
{
  "bbox": [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969],
  "checkpoint": "https://api-prod.fair.hotosm.org/api/v1/workspace/download/ramp/baseline.tflite",
  "zoom_level": 20,
  "source": "https://apps.kontur.io/raster-tiler/oam/mosaic/{z}/{x}/{y}.png",
  "confidence": 50,
  "tolerance": 0.5,
  "area_threshold": 2,
  "orthogonalize": true,
  "task": "segmentation"
}
```

