import logging
import os
import time
import uuid
from collections.abc import Iterator
from enum import Enum
from glob import glob
from pathlib import Path

import numpy as np
from geomltoolkits import georeference_prediction_tiles

from .utils import open_images_keras, open_images_pillow, remove_files, save_mask
from .yoloseg import YOLOSeg

logger = logging.getLogger(__name__)

BATCH_SIZE = 8
IMAGE_SIZE = 256

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ModelType(str, Enum):
    TFLITE = "tflite"
    KERAS = "keras"
    YOLO = "yolo"
    ONNX = "onnx"

    @classmethod
    def from_path(cls, path: str) -> "ModelType":
        suffix = Path(path).suffix.lower()
        # .tf is a directory-based SavedModel format, also loaded via Keras
        if not suffix and os.path.isdir(path):
            return cls.KERAS
        mapping = {
            ".pt": cls.YOLO,
            ".tflite": cls.TFLITE,
            ".h5": cls.KERAS,
            ".tf": cls.KERAS,
            ".onnx": cls.ONNX,
        }
        if suffix not in mapping:
            raise RuntimeError(f"Unsupported model format: {suffix}")
        return mapping[suffix]


def _build_keras_compat_objects(keras) -> dict:
    """Build custom_objects dict for loading Keras 2.x models on Keras 3.x."""
    original_from_config = keras.layers.DepthwiseConv2D.from_config

    @classmethod  # type: ignore[misc]
    def _compat_depthwise(cls, config):
        config.pop("groups", None)
        return original_from_config.__func__(cls, config)

    class FixedDropout(keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape
            return tuple([inputs.shape[i] if s == 1 else s for i, s in enumerate(self.noise_shape)])

    return {
        "DepthwiseConv2D": type(
            "DepthwiseConv2D",
            (keras.layers.DepthwiseConv2D,),
            {"from_config": _compat_depthwise},
        ),
        "FixedDropout": FixedDropout,
    }


class _SavedModelWrapper:
    """Wraps a TFSMLayer to provide a predict() interface matching Keras models."""

    def __init__(self, layer):
        self._layer = layer

    def predict(self, inputs):
        import tensorflow as tf  # ty: ignore[unresolved-import]

        tensor = tf.constant(inputs)
        try:
            result = self._layer(tensor)
        except TypeError:
            result = self._layer([tensor])
        if isinstance(result, dict):
            result = next(iter(result.values()))
        return result.numpy()


def _load_keras_model(keras, path: str):
    """Load a Keras model with fallback for Keras 2.x saved models and SavedModel directories."""
    if os.path.isdir(path):
        logger.info("Loading SavedModel directory via TFSMLayer: %s", path)
        layer = keras.layers.TFSMLayer(path, call_endpoint="serve")
        return _SavedModelWrapper(layer)

    try:
        return keras.models.load_model(path)
    except (TypeError, ValueError):
        logger.info("Retrying with Keras 2 compatibility shim")
    with keras.utils.custom_object_scope(_build_keras_compat_objects(keras)):
        return keras.models.load_model(path)


def _batch_images(image_paths: list[str], batch_size: int = BATCH_SIZE) -> Iterator[list[str]]:
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i : i + batch_size]


def _save_raw_confidence(prediction: np.ndarray, output_path: str) -> None:
    """Save raw prediction confidence (0-1 float) as 0-255 uint8 PNG."""
    clipped = np.clip(prediction, 0.0, 1.0)
    scaled = (clipped * 255).astype(np.uint8)
    save_mask(scaled, output_path, raw=True)


def initialize_model(path: str, device: str | None = None):
    model_type = ModelType.from_path(path)

    if model_type is ModelType.YOLO:
        try:
            import torch  # ty: ignore[unresolved-import]
            from ultralytics import YOLO  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError("`.pt` requires fairpredictor[pytorch]: pip install fairpredictor[pytorch]") from e
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return YOLO(path).to(device)

    if model_type is ModelType.TFLITE:
        try:
            import ai_edge_litert.interpreter as tflite
        except ImportError:
            try:
                from tensorflow import lite as tflite  # ty: ignore[unresolved-import]
            except ImportError as e:
                raise ImportError("`.tflite` requires ai-edge-litert or tensorflow: pip install ai-edge-litert") from e
        try:
            interpreter = tflite.Interpreter(model_path=path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TFLite interpreter: {e}") from e
        interpreter.allocate_tensors()
        return interpreter

    if model_type is ModelType.KERAS:
        try:
            from tensorflow import keras  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "`.h5`/`.tf` requires fairpredictor[tensorflow]: pip install fairpredictor[tensorflow]"
            ) from e
        return _load_keras_model(keras, path)

    if model_type is ModelType.ONNX:
        try:
            import onnxruntime  # noqa: F401
        except ImportError as e:
            raise ImportError("`.onnx` requires onnxruntime: pip install onnxruntime") from e
        return path

    raise RuntimeError(f"Unsupported model type: {model_type}")


def predict_tflite(interpreter, image_paths: list[str], prediction_path: str, confidence: float) -> None:
    interpreter.resize_tensor_input(interpreter.get_input_details()[0]["index"], (BATCH_SIZE, 256, 256, 3))
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output_tensor_index = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    for batch in _batch_images(image_paths):
        if len(batch) != BATCH_SIZE:
            interpreter.resize_tensor_input(interpreter.get_input_details()[0]["index"], (len(batch), 256, 256, 3))
            interpreter.allocate_tensors()
            input_tensor_index = interpreter.get_input_details()[0]["index"]
            output_tensor_index = interpreter.tensor(interpreter.get_output_details()[0]["index"])

        images = open_images_pillow(batch)
        images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
        interpreter.set_tensor(input_tensor_index, images)
        interpreter.invoke()
        preds = output_tensor_index().copy()

        target_preds = preds[..., 1]
        for idx, path in enumerate(batch):
            _save_raw_confidence(target_preds[idx], f"{prediction_path}/{Path(path).stem}.png")


def predict_keras(model, image_paths: list[str], prediction_path: str, confidence: float) -> None:
    for batch in _batch_images(image_paths):
        images = open_images_keras(batch)
        images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
        preds = model.predict(images)

        target_preds = preds[..., 1]
        for idx, path in enumerate(batch):
            _save_raw_confidence(target_preds[idx], f"{prediction_path}/{Path(path).stem}.png")


def predict_yolo(model, image_paths: list[str], prediction_path: str, confidence: float) -> None:
    for batch in _batch_images(image_paths):
        for i, r in enumerate(model.predict(batch, conf=confidence, imgsz=IMAGE_SIZE, verbose=False)):
            mask_path = f"{prediction_path}/{Path(batch[i]).stem}.png"
            if hasattr(r, "masks") and r.masks is not None:
                raw_mask = r.masks.data.max(dim=0)[0].detach().cpu().numpy()
                _save_raw_confidence(raw_mask, mask_path)
            else:
                _save_raw_confidence(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32), mask_path)


def predict_onnx(model_path: str, image_paths: list[str], prediction_path: str, confidence: float = 0.25) -> None:
    import cv2

    yoloseg = YOLOSeg(model_path, conf_thres=confidence, iou_thres=0.3)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Failed to read image: %s", image_path)
            continue
        _boxes, scores, _class_ids, masks = yoloseg(image)
        mask_path = f"{prediction_path}/{Path(image_path).stem}.png"

        if len(masks) > 0:
            max_score = float(np.max(scores)) if len(scores) > 0 else 1.0
            combined_mask = masks.max(axis=0).astype(np.float32) * max_score
            _save_raw_confidence(combined_mask, mask_path)
        else:
            _save_raw_confidence(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32), mask_path)


_PREDICT_FUNCTIONS = {
    ModelType.TFLITE: predict_tflite,
    ModelType.KERAS: predict_keras,
    ModelType.YOLO: predict_yolo,
    ModelType.ONNX: predict_onnx,
}


def run_prediction(
    checkpoint_path: str,
    input_path: str,
    prediction_path: str | None = None,
    confidence: float = 0.5,
    crs: str = "3857",
) -> str:
    if prediction_path is None:
        temp_dir = os.path.join("/tmp", "prediction", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        prediction_path = temp_dir

    start = time.time()
    logger.info("Loading model: %s", checkpoint_path)

    model_type = ModelType.from_path(checkpoint_path)
    model = initialize_model(checkpoint_path)

    logger.info("Model loaded in %d sec", round(time.time() - start))
    start = time.time()

    os.makedirs(prediction_path, exist_ok=True)
    image_paths = glob(f"{input_path}/*.tif") + glob(f"{input_path}/*.png")
    if len(image_paths) == 0:
        raise RuntimeError("No images found in the input directory")

    predict_fn = _PREDICT_FUNCTIONS[model_type]
    predict_fn(model, image_paths, prediction_path, confidence)

    logger.info("Prediction completed in %d sec (confidence=%.2f)", round(time.time() - start), confidence)

    if model_type is ModelType.KERAS:
        from tensorflow import keras  # ty: ignore[unresolved-import]

        keras.backend.clear_session()
        del model

    start = time.time()
    georeference_path = os.path.join(prediction_path, "georeference")
    georeference_prediction_tiles(prediction_path, georeference_path, overlap_pixels=3, crs=crs)
    logger.info("Georeferencing completed in %d sec", round(time.time() - start))

    remove_files(f"{prediction_path}/*.xml")
    remove_files(f"{prediction_path}/*.png")
    return georeference_path
