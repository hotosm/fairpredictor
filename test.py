import asyncio

from predictor import DEFAULT_OAM_TMS_MOSAIC, DEFAULT_RAMP_MODEL, predict

bbox = [39.231538732285344, -6.809193620326397, 39.23551086950809, -6.808174352344608]
model_link = "https://api-prod.fair.hotosm.org/api/v1/workspace/download/training_309/checkpoint.onnx"

print("Running prediction...")

asyncio.run(
    predict(
        bbox=bbox,
        model_path=model_link,
        tms_url=DEFAULT_OAM_TMS_MOSAIC,
        zoom_level=20,
        orthogonalize=True,
        debug=False,
        confidence=0.5,
    )
)
