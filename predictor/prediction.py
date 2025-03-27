# Standard library imports
import os
import time
import uuid
from glob import glob
from pathlib import Path

# Third party imports
import numpy as np

from .utils import (
    georeference_prediction_tiles,
    open_images_keras,
    open_images_pillow,
    remove_files,
    save_mask,
)
from .yoloseg import YOLOSeg

BATCH_SIZE = 8
IMAGE_SIZE = 256
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_model_type(path):
    if path.endswith(".pt"):
        return "yolo"
    elif path.endswith(".tflite"):
        return "tflite"
    elif path.endswith(".h5") or path.endswith(".tf"):
        return "keras"
    elif path.endswith(".onnx"):
        return "onnx"
    else:
        raise RuntimeError("Model type not supported")


def initialize_model(path, device=None):
    """Loads either keras, tflite, yolo, or onnx model."""
    model_type = get_model_type(path)

    if model_type == "yolo":
        try:
            import torch
            from ultralytics import YOLO
        except ImportError:  # YOLO is not installed
            raise ImportError("YOLO & torch is not installed.")
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(path).to(device)
    elif model_type == "tflite":
        try:
            import ai_edge_litert.interpreter as tflite

        except ImportError:
            print("TFlite_runtime is not installed.")
            try:
                from tensorflow import keras, lite
            except ImportError:
                raise ImportError(
                    "Install either tensorflow or tflite_runtime  to load  tflite"
                )
        try:
            interpreter = tflite.Interpreter(model_path=path)
        except ImportError:
            try:
                interpreter = lite.Interpreter(model_path=path)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize TFLite interpreter: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading TFLite model: {str(e)}")
        interpreter.allocate_tensors()
        return interpreter
    elif model_type == "keras":
        try:
            from tensorflow import keras
        except ImportError:
            raise ImportError(
                "Tensorflow is not installed, Predictions with .h5 or .tf won't work"
            )
        model = keras.models.load_model(path)
    elif model_type == "onnx":
        try:
            # from ultralytics import YOLO
            import onnxruntime
        except ImportError:  # YOLO is not installed
            raise ImportError("onnnxruntime is not installed.")
        model = path

    return model


import cv2
from scipy import ndimage
from skimage.morphology import skeletonize


def predict_tflite(interpreter, image_paths, prediction_path, confidence):
    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"], (BATCH_SIZE, 256, 256, 3)
    )
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output_tensor_index = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )
    for i in range((len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE):
        image_batch = image_paths[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
        if len(image_batch) != BATCH_SIZE:
            interpreter.resize_tensor_input(
                interpreter.get_input_details()[0]["index"],
                (len(image_batch), 256, 256, 3),
            )
            interpreter.allocate_tensors()
            input_tensor_index = interpreter.get_input_details()[0]["index"]
            output_tensor_index = interpreter.tensor(
                interpreter.get_output_details()[0]["index"]
            )
        images = open_images_pillow(image_batch)
        images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
        interpreter.set_tensor(input_tensor_index, images)
        interpreter.invoke()
        preds = output_tensor_index().copy()

        # num_classes = preds.shape[-1]
        # print(f"Model returns {num_classes} classes")
        target_class = 1
        target_preds = preds[..., target_class]

        for idx, path in enumerate(image_batch):
            confidence_map = target_preds[idx]

            # Create visualization directory
            viz_path = os.path.join(prediction_path, "visualization")
            os.makedirs(viz_path, exist_ok=True)

            # Save the raw confidence map for visualization
            confidence_viz = np.uint8(confidence_map * 255)
            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_1_confidence.png", confidence_viz
            )

            # initial_mask = confidence_map
            # # Initial mask - threshold at 0.2
            initial_mask = (confidence_map > 0.7).astype(np.uint8)
            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_2_initial_mask.png", initial_mask * 255
            )

            # 2. Try watershed segmentation for better building separation ref https://www.geeksforgeeks.org/image-segmentation-with-watershed-algorithm-opencv-python/
            # Create markers for watershed
            dist_transform = cv2.distanceTransform(initial_mask, cv2.DIST_L2, 5)
            dist_transform_viz = cv2.normalize(
                dist_transform, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_2a_distance_transform.png",
                dist_transform_viz,
            )

            # Find sure foreground areas (building centers)
            _, sure_fg = cv2.threshold(
                dist_transform, 0.5 * dist_transform.max(), 255, 0
            )
            sure_fg = sure_fg.astype(np.uint8)
            cv2.imwrite(f"{viz_path}/{Path(path).stem}_2b_sure_foreground.png", sure_fg)

            # Mark each building center as a separate marker for watershed
            markers, num_markers = ndimage.label(sure_fg)
            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_2c_markers.png",
                cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            )

            # 3. Only after watershed or as an alternative to watershed, apply erosion
            kernel = np.ones((2, 2), np.uint8)
            eroded_mask = cv2.erode(initial_mask, kernel, iterations=1)
            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_3_eroded_mask.png", eroded_mask * 255
            )

            # 4. Try using gradient information to identify building boundaries
            # This helps separate buildings that are close to each other
            sobelx = cv2.Sobel(confidence_map, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(confidence_map, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

            # Normalize and visualize gradient
            gradient_viz = cv2.normalize(
                gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            cv2.imwrite(f"{viz_path}/{Path(path).stem}_3a_gradient.png", gradient_viz)

            # 5. Use the gradient to create a mask that separates buildings
            # High gradient areas are likely building boundaries
            gradient_mask = gradient_magnitude > (0.3 * gradient_magnitude.max())
            separation_mask = eroded_mask.copy()
            separation_mask[gradient_mask] = 0
            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_3b_separation_mask.png",
                separation_mask * 255,
            )

            # Skeletonize the mask to find thin connections
            skeleton = skeletonize(separation_mask).astype(np.uint8)
            cv2.imwrite(f"{viz_path}/{Path(path).stem}_3c_skeleton.png", skeleton * 255)

            # Apply distance transform again to the separated mask
            dist_transform2 = cv2.distanceTransform(separation_mask, cv2.DIST_L2, 5)

            # Find the distance values along the skeleton
            # Low values indicate narrow connections (chicken necks)
            skeleton_distances = dist_transform2 * skeleton

            # Find potential cut points - locations where skeleton is thin
            # (small distance values along the skeleton)
            cut_threshold = 3  # Adjust this threshold for your data
            cut_points = (skeleton_distances > 0) & (skeleton_distances < cut_threshold)

            # Create a new mask with the narrow connections removed
            enhanced_separation_mask = separation_mask.copy()
            enhanced_separation_mask[cut_points] = 0

            # Visualize the cut points and enhanced separation
            cut_points_viz = np.zeros_like(separation_mask, dtype=np.uint8)
            cut_points_viz[cut_points] = 255
            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_3d_cut_points.png", cut_points_viz
            )

            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_3e_enhanced_separation.png",
                enhanced_separation_mask * 255,
            )

            # Now use this enhanced mask for component labeling
            labeled_mask, num_components = ndimage.label(enhanced_separation_mask)

            # 6. Now do component labeling on this enhanced mask
            # labeled_mask, num_components = ndimage.label(separation_mask)
            # Apply component labeling
            # labeled_mask, num_components = ndimage.label(eroded_mask)

            # Create a colorized version of the labeled mask for visualization
            # This gives each component a different color
            colored_labels = np.zeros(
                (labeled_mask.shape[0], labeled_mask.shape[1], 3), dtype=np.uint8
            )

            # Create colorful visualization of components
            unique_labels = np.unique(labeled_mask)[1:]  # Skip background (0)
            for label in unique_labels:
                # Generate a random color for this component
                color = np.random.randint(0, 255, size=3)
                colored_labels[labeled_mask == label] = color

            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_4_labeled_components.png", colored_labels
            )

            # Continue with your existing component processing
            refined_mask = np.zeros_like(initial_mask)

            for comp_id in range(1, num_components + 1):
                component_mask = labeled_mask == comp_id
                avg_confidence = np.mean(confidence_map[component_mask])

                if avg_confidence >= confidence:
                    refined_mask[component_mask] = 1

            # Save the refined mask after confidence filtering
            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_5_refined_mask.png", refined_mask * 255
            )

            # Apply final morphological operations
            kernel = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(
                f"{viz_path}/{Path(path).stem}_6_final_mask.png", cleaned_mask * 255
            )

            # Save the final mask for actual prediction output
            save_mask(
                cleaned_mask,
                str(f"{prediction_path}/{Path(path).stem}.png"),
            )


def predict_keras(model, image_paths, prediction_path, confidence):

    for i in range((len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE):
        image_batch = image_paths[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
        images = open_images_keras(image_batch)
        images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
        preds = model.predict(images)
        num_classes = preds.shape[-1]
        print(f"Model returns {num_classes} classes")
        target_class = 1
        target_preds = preds[..., target_class]
        binary_masks = np.where(target_preds > confidence, 1, 0)
        binary_masks = np.expand_dims(binary_masks, axis=-1)

        for idx, path in enumerate(image_batch):
            save_mask(
                binary_masks[idx],
                str(f"{prediction_path}/{Path(path).stem}.png"),
            )


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


def predict_onnx(model_path, image_paths, prediction_path, confidence=0.25):
    import cv2
    from PIL import Image

    yoloseg = YOLOSeg(model_path, conf_thres=confidence, iou_thres=0.3)

    # Iterate through all images
    for image_path in image_paths:
        image = cv2.imread(image_path)
        boxes, scores, class_ids, masks = yoloseg(image)
        mask_path = f"{prediction_path}/{Path(image_path).stem}.png"

        if len(masks) > 0:
            combined_mask = masks.max(axis=0) * 255  # Combine masks and scale to 255
            result = Image.fromarray(combined_mask.astype(np.uint8))
            result.save(mask_path)
        else:
            preds = np.zeros(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                dtype=np.float32,
            )
            save_mask(preds, mask_path)


def save_predictions(preds, image_batch, prediction_path):
    for idx, path in enumerate(image_batch):
        save_mask(preds[idx], str(f"{prediction_path}/{Path(path).stem}.png"))


def run_prediction(
    checkpoint_path: str,
    input_path: str,
    prediction_path: str = None,
    confidence: float = 0.5,
    crs: str = "3857",
) -> None:
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
    image_paths = glob(f"{input_path}/*.tif")
    if len(image_paths) == 0:
        raise RuntimeError("No images found in the input directory")

    if model_type == "tflite":

        predict_tflite(model, image_paths, prediction_path, confidence)

    elif model_type == "keras":
        predict_keras(model, image_batch, confidence)

    elif model_type == "yolo":
        predict_yolo(model, image_paths, prediction_path, confidence)
    elif model_type == "onnx":
        predict_onnx(model, image_paths, prediction_path, confidence)

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
    georeference_prediction_tiles(
        prediction_path, georeference_path, overlap_pixels=2, crs=crs
    )
    print(f"It took {round(time.time()-start)} sec to georeference")

    remove_files(f"{prediction_path}/*.xml")
    remove_files(f"{prediction_path}/*.png")
    return georeference_path
