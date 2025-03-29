import asyncio

from predictor import DEFAULT_OAM_TMS_MOSAIC, DEFAULT_RAMP_MODEL, predict

bbox = [100.56228, 13.685230, 100.56383, 13.685961]
asyncio.run(
    predict(
        bbox=bbox,
        model_path=DEFAULT_RAMP_MODEL,
        tms_url=DEFAULT_OAM_TMS_MOSAIC,
        zoom_level=20,
        orthogonalize=True,
        remove_metadata=False,
        confidence=0.5,
    )
)
