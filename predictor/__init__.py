from .app import predict

DEFAULT_OAM_TMS_MOSAIC = (
    "https://apps.kontur.io/raster-tiler/oam/mosaic/{z}/{x}/{y}.png"
)
DEFAULT_YOLO_MODEL_V1 = "https://api-prod.fair.hotosm.org/api/v1/workspace/download/yolo/yolov8s_v1-seg.onnx"
DEFAULT_RAMP_MODEL = (
    "https://api-prod.fair.hotosm.org/api/v1/workspace/download/ramp/baseline.tflite"
)
DEFAULT_YOLO_MODEL_V2 = "https://api-prod.fair.hotosm.org/api/v1/workspace/download/yolo/yolov8s_v2-seg.onnx"
