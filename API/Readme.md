## fAIr Prediction API

FastAPI-based HTTP API for fAIr model predictions, served from the project root (`main.py`).

### Prerequisites

- Docker

### Getting Started

1. Clone the repo

    ```bash
    git clone https://github.com/hotosm/fairpredictor.git
    cd fairpredictor
    ```

2. Build the image

    ```bash
    docker build -t fairpredictor .
    ```

3. Run the container

    ```bash
    docker run --rm -p 8000:8000 fairpredictor
    ```

4. API documentation

    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc

### POST /predict/

Required fields: `bbox`, `checkpoint`, `zoom_level`, `source`.

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

#### All fields

| Field | Type | Default | Description |
|---|---|---|---|
| `bbox` | `float[4]` | required | `[min_lon, min_lat, max_lon, max_lat]` |
| `checkpoint` | `str` | required | URL or local path to model file |
| `zoom_level` | `int` | required | Tile zoom level (18-22) |
| `source` | `str` | required | TMS URL template `{z}/{x}/{y}` |
| `task` | `str` | `"segmentation"` | Task type (`segmentation` only for now) |
| `confidence` | `int` | `50` | Confidence threshold % (0-100) |
| `tolerance` | `float` | `0.5` | Polygon simplification tolerance (0-10) |
| `area_threshold` | `float` | `2` | Minimum polygon area (0-20) |
| `orthogonalize` | `bool` | `true` | Apply orthogonalization |
| `ortho_skew_tolerance_deg` | `int` | `15` | Skew tolerance for orthogonalization (0-45) |
| `ortho_max_angle_change_deg` | `int` | `15` | Max angle change for orthogonalization (0-45) |
| `get_predictions_as_points` | `bool` | `false` | Include centroid points in output |
| `make_geoms_valid` | `bool` | `true` | Validate and fix output polygons |
| `output_path` | `str` | `null` | Directory to save outputs |

