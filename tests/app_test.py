import os
import shutil

import requests
from efficientnet.tfkeras import EfficientNetB4

from predictor import predict

model_path = "checkpoint.h5"
bbox = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
zoom_level = 20
tms_url = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"

if not os.path.exists(model_path):
    url = "https://fair-dev.hotosm.org/api/v1/workspace/download/dataset_65/output/training_297/checkpoint.h5"
    response = requests.get(url, stream=True)
    with open(model_path, "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)


bbox = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
zoom_level = 20
tms_url = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"

my_predictions = predict(bbox, model_path, zoom_level, tms_url)
print(my_predictions)
