import json
import os
import shutil

from predictor import predict

# import efficientnet.keras as efn


# import requests


# model_path = "checkpoint.tflite"
# bbox = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
# zoom_level = 20
# tms_url = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"


# bbox = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
# zoom_level = 20
# tms_url = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"

# my_predictions = predict(bbox, model_path, zoom_level, tms_url, remove_metadata=False)
# print(my_predictions)


model_path = "checkpoints/yolo/checkpoint.onnx"
bbox = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
zoom_level = 20
tms_url = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"


my_predictions = predict(bbox, model_path, zoom_level, tms_url, remove_metadata=False)
with open("predictions.geojson", "w") as f:
    json.dump(my_predictions, f)
