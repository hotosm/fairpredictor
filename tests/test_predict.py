import asyncio
import os

import pytest

from predictor import predict

TMS_URL = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"
BBOX = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
FAIR_BASE_URL = "https://fair-dev.hotosm.org/api/v1/workspace/download"
DATASET_ID = "dataset_65"
TRAINING_ID = "training_457"


@pytest.fixture
def model_paths():
    base_path = os.path.abspath(os.path.dirname(__file__))
    return {
        "h5": os.path.join(base_path, "checkpoints", "ramp", "checkpoint.h5"),
        "tflite": os.path.join(base_path, "checkpoints", "ramp", "checkpoint.tflite"),
        "pt": os.path.join(base_path, "checkpoints", "yolo", "checkpoint.pt"),
        "onnx": os.path.join(base_path, "checkpoints", "yolo", "checkpoint.onnx"),
    }


@pytest.fixture
def zoom_level():
    return 20


# @pytest.mark.skip(reason="h5 model test is disabled")
# def test_predict_h5(model_paths, zoom_level):
#     predictions = predict(BBOX, model_paths["h5"], zoom_level, TMS_URL)
#     assert isinstance(predictions, dict)
#     assert len(predictions["features"]) > 0


def test_predict_tflite(model_paths, zoom_level):
    predictions = asyncio.run(
        predict(
            bbox=BBOX,
            model_path=model_paths["tflite"],
            zoom_level=zoom_level,
            tms_url=TMS_URL,
        )
    )
    assert isinstance(predictions, dict)
    assert len(predictions["features"]) > 0


# @pytest.mark.skip(reason="pt model test is disabled")
# def test_predict_pt(model_paths, zoom_level):
#     predictions = predict(BBOX, model_paths["pt"], zoom_level, TMS_URL)
#     assert isinstance(predictions, dict)
#     assert len(predictions["features"]) > 0


def test_predict_onnx(model_paths, zoom_level):
    predictions = asyncio.run(
        predict(
            bbox=BBOX,
            model_path=model_paths["onnx"],
            zoom_level=zoom_level,
            tms_url=TMS_URL,
        )
    )
    assert isinstance(predictions, dict)
    assert len(predictions["features"]) > 0
