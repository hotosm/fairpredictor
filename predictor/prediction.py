# Standard library imports
import concurrent.futures
import os
import time
import uuid
from glob import glob
from pathlib import Path

# Third party imports
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print(
        "TFlite_runtime is not installed , Predictions with .tflite extension won't work"
    )
try:
    from tensorflow import keras
except ImportError:
    print("Tensorflow is not installed , Predictions with .h5 or .tf won't work")


from .georeferencer import georeference
from .utils import open_images_keras, open_images_pillow, remove_files, save_mask

BATCH_SIZE = 8
IMAGE_SIZE = 256
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_prediction(
    checkpoint_path: str,
    input_path: str,
    prediction_path: str = None,
    confidence: float = 0.5,
    tile_overlap_distance: float = 0.15,
) -> None:
    """Predict building footprints for aerial images given a model checkpoint.

    This function reads the model weights from the checkpoint path and outputs
    predictions in GeoTIF format. The input images have to be in PNG format.

    The predicted masks will be georeferenced with EPSG:3857 as CRS.

    Args:
        checkpoint_path: Path where the weights of the model can be found.
        input_path: Path of the directory where the images are stored.
        prediction_path: Path of the directory where the predicted images will go.
        confidence: Threshold probability for filtering out low-confidence predictions.
        tile_overlap_distance : Provides tile overlap distance to remove the strip between predictions.

    Example::

        predict(
            "model_1_checkpt.tf",
            "data/inputs_v2/4",
            "data/predictions/4"
        )
    """
    if prediction_path is None:
        # Generate a temporary download path using a UUID
        temp_dir = os.path.join("/tmp", "prediction", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        prediction_path = temp_dir
    start = time.time()
    print(f"Using : {checkpoint_path}")
    if checkpoint_path.endswith(".tflite"):
        interpreter = tflite.Interpreter(model_path=checkpoint_path)
        interpreter.resize_tensor_input(
            interpreter.get_input_details()[0]["index"], (BATCH_SIZE, 256, 256, 3)
        )
        interpreter.allocate_tensors()
        input_tensor_index = interpreter.get_input_details()[0]["index"]
        output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    else:
        model = keras.models.load_model(checkpoint_path)
    print(f"It took {round(time.time()-start)} sec to load model")
    start = time.time()

    os.makedirs(prediction_path, exist_ok=True)
    image_paths = glob(f"{input_path}/*.png")
    if checkpoint_path.endswith(".tflite"):
        for i in range((len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE):
            image_batch = image_paths[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
            if len(image_batch) != BATCH_SIZE:
                interpreter.resize_tensor_input(
                    interpreter.get_input_details()[0]["index"],
                    (len(image_batch), 256, 256, 3),
                )
                interpreter.allocate_tensors()
                input_tensor_index = interpreter.get_input_details()[0]["index"]
                output = interpreter.tensor(
                    interpreter.get_output_details()[0]["index"]
                )
            images = open_images_pillow(image_batch)
            images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
            interpreter.set_tensor(input_tensor_index, images)
            interpreter.invoke()
            preds = output()
            preds = np.argmax(preds, axis=-1)
            preds = np.expand_dims(preds, axis=-1)
            preds = np.where(
                preds > confidence, 1, 0
            )  # Filter out low confidence predictions

            for idx, path in enumerate(image_batch):
                save_mask(
                    preds[idx],
                    str(f"{prediction_path}/{Path(path).stem}.png"),
                )
    else:
        for i in range((len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE):
            image_batch = image_paths[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
            images = open_images_keras(image_batch)
            images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
            preds = model.predict(images)
            preds = np.argmax(preds, axis=-1)
            preds = np.expand_dims(preds, axis=-1)
            preds = np.where(
                preds > confidence, 1, 0
            )  # Filter out low confidence predictions

            for idx, path in enumerate(image_batch):
                save_mask(
                    preds[idx],
                    str(f"{prediction_path}/{Path(path).stem}.png"),
                )

    print(
        f"It took {round(time.time()-start)} sec to predict with {confidence} Confidence Threshold"
    )
    if not checkpoint_path.endswith(".tflite"):
        keras.backend.clear_session()
        del model
    start = time.time()
    georeference_path = os.path.join(prediction_path, "georeference")
    georeference(
        prediction_path,
        georeference_path,
        is_mask=True,
        tile_overlap_distance=tile_overlap_distance,
    )
    print(f"It took {round(time.time()-start)} sec to georeference")

    remove_files(f"{prediction_path}/*.xml")
    remove_files(f"{prediction_path}/*.png")
    return georeference_path
