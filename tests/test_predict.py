import os
import shutil
import tempfile
import unittest

from predictor import predict

TMS_URL = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"
BBOX = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
FAIR_BASE_URL = "https://fair-dev.hotosm.org/api/v1/workspace/download"
DATASET_ID = "dataset_65"
TRAINING_ID = "training_457"


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.model_path_h5 = "checkpoints/ramp/checkpoint.h5"
        self.model_path_tflite = "checkpoints/ramp/checkpoint.tflite"
        self.model_path_pt = "checkpoints/yolo/checkpoint.pt"

    def test_predict_h5(self):
        zoom_level = 20
        predictions = predict(BBOX, self.model_path_h5, zoom_level, TMS_URL)
        self.assertIsInstance(predictions, dict)
        self.assertTrue(len(predictions["features"]) > 0)

    def test_predict_tflite(self):
        zoom_level = 20
        predictions = predict(BBOX, self.model_path_tflite, zoom_level, TMS_URL)
        self.assertIsInstance(predictions, dict)
        self.assertTrue(len(predictions["features"]) > 0)

    def test_predict_pt(self):
        zoom_level = 20
        predictions = predict(BBOX, self.model_path_pt, zoom_level, TMS_URL)
        self.assertIsInstance(predictions, dict)
        self.assertTrue(len(predictions["features"]) > 0)


if __name__ == "__main__":
    unittest.main()
