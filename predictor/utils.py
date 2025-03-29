import glob
import os
import re
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import requests
from geomltoolkits.utils import georeference_tile
from PIL import Image

IMAGE_SIZE = 256


def open_images_keras(paths: List[str]) -> np.ndarray:
    """Open images from some given paths."""
    images = []
    for path in paths:
        image = keras.preprocessing.image.load_img(
            path, target_size=(IMAGE_SIZE, IMAGE_SIZE)
        )
        image = np.array(image.getdata()).reshape(IMAGE_SIZE, IMAGE_SIZE, 3) / 255.0
        images.append(image)

    return np.array(images)


def open_images_pillow(paths: List[str]) -> np.ndarray:
    """Open images from given paths using Pillow and resize them."""
    images = []
    for path in paths:
        img = Image.open(path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) / 255.0
        images.append(img_array)

    return np.array(images)


def remove_files(pattern: str) -> None:
    """Remove files matching a wildcard."""
    files = glob.glob(pattern)
    for file in files:
        os.remove(file)


def save_mask(mask: np.ndarray, filename: str) -> None:
    """Save the mask array to the specified location."""
    reshaped_mask = mask.reshape((IMAGE_SIZE, IMAGE_SIZE)) * 255
    result = Image.fromarray(reshaped_mask.astype(np.uint8))
    result.save(filename)

    # with rasterio.open(
    #     filename,
    #     'w',
    #     driver='GTiff',
    #     height=IMAGE_SIZE,
    #     width=IMAGE_SIZE,
    #     count=1,
    #     dtype=rasterio.float32,
    #     nodata=0
    # ) as dst:
    #     dst.write(reshaped_mask, 1)


def georeference_prediction_tiles(
    prediction_path: str,
    georeference_path: str,
    overlap_pixels: int = 0,
    crs: str = "3857",
) -> List[str]:
    """
    Georeference all prediction tiles based on their embedded x,y,z coordinates in filenames.

    Args:
        prediction_path: Directory containing prediction tiles
        georeference_path: Directory to save georeferenced tiles
        tile_overlap_distance: Overlap distance between tiles

    Returns:
        List of paths to georeferenced tiles
    """
    os.makedirs(georeference_path, exist_ok=True)

    image_files = glob.glob(os.path.join(prediction_path, "*.png"))
    image_files.extend(glob.glob(os.path.join(prediction_path, "*.jpeg")))

    georeferenced_files = []

    for image_file in image_files:
        filename = os.path.basename(image_file)
        filename_without_ext = re.sub(r"\.(png|jpeg)$", "", filename)

        try:
            parts = re.split("-", filename_without_ext)
            if len(parts) >= 3:
                # Get the last three parts which should be x, y, z
                x_tile, y_tile, zoom = map(int, parts[-3:])

                output_tiff = os.path.join(
                    georeference_path, f"{filename_without_ext}.tif"
                )

                georeferenced_file = georeference_tile(
                    input_tiff=image_file,
                    x=x_tile,
                    y=y_tile,
                    z=zoom,
                    output_tiff=output_tiff,
                    crs=crs,
                    overlap_pixels=overlap_pixels,
                )

                georeferenced_files.append(georeferenced_file)
                # print(f"Georeferenced {filename} to {output_tiff}")
            else:
                print(f"Warning: Could not extract tile coordinates from {filename}")

        except Exception as e:
            print(f"Error georeferencing {filename}: {str(e)}")

    print(f"Georeferenced {len(georeferenced_files)} tiles to {georeference_path}")
    return georeference_path


def download_or_validate_model(model_path: str) -> str:
    """
    Download model from URL or validate local model path.

    Args:
        model_path: URL or local path to model file

    Returns:
        Path to local model file (downloaded or original)

    Raises:
        RuntimeError: If model download fails
        FileNotFoundError: If local model file doesn't exist
    """
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
            raise RuntimeError(f"Failed to download model from {model_path}: {str(e)}")

    elif isinstance(model_path, str) and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_path


def clean_building_mask(
    target_preds: np.ndarray,
    confidence_threshold=0.5,
    morph_size=3,
):
    """
    Clean up building masks to remove thin connections and improve precision.

    Args:
        target_preds: Raw prediction or binary mask (0-1 range)
        confidence_threshold: Base threshold for building/non-building (ignored if input is already binary)
        morph_size: Size of morphological operation kernel
    Returns:
        Cleaned binary mask
    """
    # Check if input is already binary (only contains 0s and 1s)
    is_binary = np.array_equal(
        target_preds, target_preds.astype(bool).astype(target_preds.dtype)
    )

    if is_binary:
        # Skip thresholding if input is already binary
        binary_mask = target_preds.astype(np.uint8)
    else:
        binary_mask = np.where(target_preds > confidence_threshold, 1, 0).astype(
            np.uint8
        )

    # Define kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))

    # Apply opening to remove thin connections (erode then dilate)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Final erosion to amplify differences between buildings
    eroded_mask = cv2.erode(opened_mask, kernel, iterations=1)

    # Fill holes in buildings with closing
    filled_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_CLOSE, kernel)

    return filled_mask
