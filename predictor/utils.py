import glob
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_SIZE = 256


def open_images_keras(paths: list[str]) -> np.ndarray:
    from tensorflow import keras  # ty: ignore[unresolved-import]

    images = []
    for path in paths:
        image = keras.preprocessing.image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image).reshape(IMAGE_SIZE, IMAGE_SIZE, 3) / 255.0
        images.append(image)

    return np.array(images)


def open_images_pillow(paths: list[str]) -> np.ndarray:
    images = []
    for path in paths:
        img = Image.open(path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) / 255.0
        images.append(img_array)

    return np.array(images)


def remove_files(pattern: str) -> None:
    files = glob.glob(pattern)
    for file in files:
        os.remove(file)


def save_mask(mask: np.ndarray, filename: str, *, raw: bool = False) -> None:
    reshaped = mask.reshape((IMAGE_SIZE, IMAGE_SIZE))
    if not raw:
        reshaped = reshaped * 255
    result = Image.fromarray(reshaped.astype(np.uint8))
    result.save(filename)


def download_or_validate_model(model_path: str) -> str:
    if isinstance(model_path, str) and model_path.startswith(("http://", "https://")):
        try:
            response = requests.head(model_path, timeout=10)
            response.raise_for_status()

            file_ext = Path(model_path).suffix.lower()

            if not file_ext:
                content_type = response.headers.get("Content-Type", "")
                if "tflite" in model_path.lower() or "tflite" in content_type:
                    file_ext = ".tflite"
                elif "onnx" in model_path.lower() or "onnx" in content_type:
                    file_ext = ".onnx"
                elif "h5" in model_path.lower() or "keras" in model_path.lower():
                    file_ext = ".h5"
                else:
                    file_ext = ".tflite"

            response = requests.get(model_path, timeout=30)
            response.raise_for_status()

            _, temp_file_path = tempfile.mkstemp(suffix=file_ext)

            with open(temp_file_path, "wb") as f:
                f.write(response.content)

            return temp_file_path
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {model_path}: {e}") from e

    elif isinstance(model_path, str) and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_path


def threshold_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.where(mask > threshold, 1, 0).astype(np.uint8)
