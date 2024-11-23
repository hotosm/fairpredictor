# Standard library imports
import os
import time
import uuid
from glob import glob
from pathlib import Path

# Third party imports
import numpy as np

from .georeferencer import georeference
from .utils import open_images_keras, open_images_pillow, remove_files, save_mask

BATCH_SIZE = 8
IMAGE_SIZE = 256
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["YOLO_AUTOINSTALL"] = "false"


def get_model_type(path):
    if path.endswith(".pt"):
        return "yolo"
    elif path.endswith(".tflite"):
        return "tflite"
    elif path.endswith(".h5") or path.endswith(".tf"):
        return "keras"
    else:
        raise RuntimeError("Model type not supported")


def initialize_model(path):
    """Loads either keras, tflite, or yolo model."""
    model_type = get_model_type(path)

    if model_type == "yolo":
        try:
            import torch
            from ultralytics import YOLO
        except ImportError:  # YOLO is not installed
            raise ImportError("YOLO & torch is not installed.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(path).to(device)
    elif model_type == "tflite":
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            raise ImportError("TFlite_runtime is not installed.")
        model = load_tflite_model(path)
    elif model_type == "keras":
        try:
            from tensorflow import keras
        except ImportError:
            raise ImportError(
                "Tensorflow is not installed, Predictions with .h5 or .tf won't work"
            )
        model = keras.models.load_model(path)
    else:
        return path
    return model


def load_tflite_model(checkpoint_path):
    try:
        interpreter = tflite.Interpreter(model_path=checkpoint_path)
    except Exception as ex:
        interpreter = lite.Interpreter(model_path=checkpoint_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_tflite(interpreter, image_batch, confidence):
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    images = open_images_pillow(image_batch)
    images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
    interpreter.set_tensor(input_tensor_index, images)
    interpreter.invoke()
    preds = output()
    preds = np.argmax(preds, axis=-1)
    preds = np.expand_dims(preds, axis=-1)
    preds = np.where(preds > confidence, 1, 0)
    return preds


def predict_keras(model, image_batch, confidence):
    images = open_images_keras(image_batch)
    images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    preds = model.predict(images)
    preds = np.argmax(preds, axis=-1)
    preds = np.expand_dims(preds, axis=-1)
    preds = np.where(preds > confidence, 1, 0)
    return preds


def predict_yolo(model, image_paths, prediction_path, confidence):
    for idx in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[idx : idx + BATCH_SIZE]
        for i, r in enumerate(
            model.predict(batch, conf=confidence, imgsz=IMAGE_SIZE, verbose=False)
        ):
            if hasattr(r, "masks") and r.masks is not None:
                preds = (
                    r.masks.data.max(dim=0)[0].detach().cpu().numpy()
                )  # Combine masks and convert to numpy
            else:
                preds = np.zeros(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                    ),
                    dtype=np.float32,
                )  # Default if no masks
            save_mask(preds, str(f"{prediction_path}/{Path(batch[i]).stem}.png"))


def save_predictions(preds, image_batch, prediction_path):
    for idx, path in enumerate(image_batch):
        save_mask(preds[idx], str(f"{prediction_path}/{Path(path).stem}.png"))


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
        temp_dir = os.path.join("/tmp", "prediction", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        prediction_path = temp_dir

    start = time.time()
    print(f"Using : {checkpoint_path}")

    model_type = get_model_type(checkpoint_path)
    model = initialize_model(checkpoint_path)

    print(f"It took {round(time.time()-start)} sec to load model")
    start = time.time()

    os.makedirs(prediction_path, exist_ok=True)
    image_paths = glob(f"{input_path}/*.png")

    if model_type == "tflite":
        for i in range((len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE):
            image_batch = image_paths[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
            if len(image_batch) != BATCH_SIZE:
                model.resize_tensor_input(
                    model.get_input_details()[0]["index"],
                    (len(image_batch), 256, 256, 3),
                )
                model.allocate_tensors()
            preds = predict_tflite(model, image_batch, confidence)
            save_predictions(preds, image_batch, prediction_path)
    elif model_type == "keras":
        for i in range((len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE):
            image_batch = image_paths[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
            preds = predict_keras(model, image_batch, confidence)
            save_predictions(preds, image_batch, prediction_path)
    elif model_type == "yolo":
        predict_yolo(model, image_paths, prediction_path, confidence)
    else:
        raise RuntimeError("Loaded model is not supported")

    print(
        f"It took {round(time.time()-start)} sec to predict with {confidence} Confidence Threshold"
    )

    if model_type == "keras":
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
