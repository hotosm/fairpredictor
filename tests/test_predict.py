import os
import shutil
import tempfile
import unittest

import efficientnet.tfkeras as efn
import requests

from predictor import predict

# Global variables
TMS_URL = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"
BBOX = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
FAIR_BASE_URL = "https://fair-dev.hotosm.org/api/v1/workspace/download"
DATASET_ID = "dataset_65"
TRAINING_ID = "training_297"


class TestPredictor(unittest.TestCase):
    def setUp(self):
        model_url = f"{FAIR_BASE_URL}/{DATASET_ID}/output/{TRAINING_ID}/checkpoint.h5"
        self.model_path = tempfile.NamedTemporaryFile(suffix=".h5").name
        response = requests.get(model_url, stream=True)
        with open(self.model_path, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)

    def tearDown(self):
        if self.model_path:
            try:
                os.remove(self.model_path)
            except OSError:
                pass

    def test_predict(self):
        zoom_level = 20
        predictions = predict(BBOX, self.model_path, zoom_level, TMS_URL)
        self.assertIsInstance(predictions, dict)
        self.assertTrue(len(predictions["features"]) > 0)


if __name__ == "__main__":
    unittest.main()
