import os
import shutil

import efficientnet.keras as efn

from predictor import predict

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


model_path = "checkpoints/ramp/checkpoint.h5"
bbox = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
zoom_level = 20
tms_url = "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"


my_predictions = predict(bbox, model_path, zoom_level, tms_url, remove_metadata=False)
print(my_predictions)
