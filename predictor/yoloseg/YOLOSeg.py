import math
from typing import Any

import cv2
import numpy as np
import onnxruntime

from .utils import nms, sigmoid, xywh2xyxy


class YOLOSeg:
    def __init__(self, path: str, conf_thres: float = 0.7, iou_thres: float = 0.5, num_masks: int = 32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self._initialize_session(path)

    def __call__(self, image: np.ndarray) -> tuple[Any, Any, Any, Any]:
        return self.segment_objects(image)

    def _initialize_session(self, path: str) -> None:
        self.session = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in model_inputs]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]

    def segment_objects(self, image: np.ndarray) -> tuple[Any, Any, Any, Any]:
        self.img_height, self.img_width = image.shape[:2]
        input_tensor = self._prepare_input(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        self.boxes, self.scores, self.class_ids, mask_pred = self._process_box_output(outputs[0])  # ty: ignore[invalid-argument-type]
        self.mask_maps = self._process_mask_output(mask_pred, outputs[1])  # ty: ignore[invalid-argument-type]

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        return input_img[np.newaxis, :, :, :].astype(np.float32)

    def _process_box_output(self, box_output: np.ndarray) -> tuple:
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        class_ids = np.argmax(box_predictions[:, 4:], axis=1)
        boxes = self._extract_boxes(box_predictions)
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def _process_mask_output(self, mask_predictions: np.ndarray, mask_output: np.ndarray) -> np.ndarray | list[Any]:
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)
        num_mask, mask_height, mask_width = mask_output.shape
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        scale_boxes = self._rescale_boxes(self.boxes, (self.img_height, self.img_width), (mask_height, mask_width))

        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))

        for i in range(len(scale_boxes)):
            scale_x1 = math.floor(scale_boxes[i][0])
            scale_y1 = math.floor(scale_boxes[i][1])
            scale_x2 = math.ceil(scale_boxes[i][2])
            scale_y2 = math.ceil(scale_boxes[i][3])

            x1 = math.floor(self.boxes[i][0])
            y1 = math.floor(self.boxes[i][1])
            x2 = math.ceil(self.boxes[i][2])
            y2 = math.ceil(self.boxes[i][3])

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def _extract_boxes(self, box_predictions: np.ndarray) -> np.ndarray:
        boxes = box_predictions[:, :4]
        boxes = self._rescale_boxes(boxes, (self.input_height, self.input_width), (self.img_height, self.img_width))
        boxes = xywh2xyxy(boxes)

        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    @staticmethod
    def _rescale_boxes(boxes: np.ndarray, input_shape: tuple[int, int], image_shape: tuple[int, int]) -> np.ndarray:
        input_shape_arr = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape_arr, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes
