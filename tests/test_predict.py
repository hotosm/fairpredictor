import asyncio
import os

import pytest

from predictor import predict

TMS_URL = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"
BBOX = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
ZOOM_LEVEL = 20


@pytest.fixture
def model_paths():
    base_path = os.path.abspath(os.path.dirname(__file__))
    return {
        "h5": os.path.join(base_path, "checkpoints", "ramp", "checkpoint.h5"),
        "tflite": os.path.join(base_path, "checkpoints", "ramp", "checkpoint.tflite"),
        "pt": os.path.join(base_path, "checkpoints", "yolo", "yolov8n-seg.pt"),
        "onnx": os.path.join(base_path, "checkpoints", "yolo", "checkpoint.onnx"),
    }


@pytest.fixture
def tf_model_path(model_paths):
    """Convert h5 checkpoint to SavedModel (.tf) format for testing."""
    from tensorflow import keras  # ty: ignore[unresolved-import]

    tf_dir = os.path.join(os.path.dirname(model_paths["h5"]), "checkpoint_savedmodel")
    if os.path.isdir(tf_dir):
        return tf_dir

    from predictor.prediction import _load_keras_model

    model = _load_keras_model(keras, model_paths["h5"])
    model.export(tf_dir)
    return tf_dir


def _run_predict(model_path: str, **kwargs) -> dict:
    return asyncio.run(
        predict(
            bbox=BBOX,
            model_path=model_path,
            zoom_level=ZOOM_LEVEL,
            tms_url=TMS_URL,
            **kwargs,
        )
    )


def _assert_valid_geojson(result: dict, expect_confidence: bool = True) -> None:
    assert isinstance(result, dict)
    assert result["type"] == "FeatureCollection"
    assert "features" in result
    assert len(result["features"]) > 0
    for feature in result["features"]:
        assert feature["type"] == "Feature"
        assert "geometry" in feature
        assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon", "Point")
        assert "properties" in feature
        if expect_confidence:
            assert "confidence" in feature["properties"]
            conf = feature["properties"]["confidence"]
            assert isinstance(conf, float)
            assert 0.0 <= conf <= 1.0


def test_predict_tflite(model_paths):
    result = _run_predict(model_paths["tflite"])
    _assert_valid_geojson(result)


def test_predict_onnx(model_paths):
    result = _run_predict(model_paths["onnx"])
    _assert_valid_geojson(result)


def test_predict_h5(model_paths):
    result = _run_predict(model_paths["h5"])
    _assert_valid_geojson(result)


def test_predict_pt(model_paths):
    result = _run_predict(model_paths["pt"])
    _assert_valid_geojson(result)


def test_predict_tf(tf_model_path):
    result = _run_predict(tf_model_path)
    _assert_valid_geojson(result)


def test_predict_tflite_no_orthogonalize(model_paths):
    result = _run_predict(model_paths["tflite"], orthogonalize=False)
    _assert_valid_geojson(result)


def test_predict_onnx_high_confidence(model_paths):
    result = _run_predict(model_paths["onnx"], confidence=0.9)
    assert isinstance(result, dict)
    assert result["type"] == "FeatureCollection"
    assert "features" in result


def test_predict_tflite_as_polygons(model_paths):
    result = _run_predict(model_paths["tflite"], get_predictions_as_points=False)
    _assert_valid_geojson(result)
    for feature in result["features"]:
        assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_predict_invalid_task(model_paths):
    with pytest.raises(NotImplementedError, match="not yet supported"):
        _run_predict(model_paths["tflite"], task="detection")


def test_predict_missing_bbox(model_paths):
    with pytest.raises(ValueError, match="Either bbox or geojson"):
        asyncio.run(
            predict(
                model_path=model_paths["tflite"],
                zoom_level=ZOOM_LEVEL,
                tms_url=TMS_URL,
                bbox=None,
                geojson=None,
            )
        )


def test_predict_invalid_confidence(model_paths):
    with pytest.raises(ValueError, match="Confidence must be between"):
        _run_predict(model_paths["tflite"], confidence=1.5)
