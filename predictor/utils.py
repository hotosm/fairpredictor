import glob
import os
import re
from typing import List

import numpy as np
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
                
                output_tiff = os.path.join(georeference_path, f"{filename_without_ext}.tif")
                
                georeferenced_file = georeference_tile(
                    input_tiff=image_file,
                    x=x_tile,
                    y=y_tile,
                    z=zoom, 
                    output_tiff=output_tiff,
                    crs=crs,
                    overlap_pixels=overlap_pixels
                )
                
                georeferenced_files.append(georeferenced_file)
                # print(f"Georeferenced {filename} to {output_tiff}")
            else:
                print(f"Warning: Could not extract tile coordinates from {filename}")
                
        except Exception as e:
            print(f"Error georeferencing {filename}: {str(e)}")
    
    print(f"Georeferenced {len(georeferenced_files)} tiles to {georeference_path}")
    return georeference_path