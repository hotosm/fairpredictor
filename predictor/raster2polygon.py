import os
import uuid

from raster2polygon import polygonize


def polygonizer(
    prediction_path,
    output_path: str = None,
    tolerance: float = 0.01,
    merging_distance_threshold: float = 0.5,
    area_threshold: float = 5,
):
    if output_path is None:
        # Generate a temporary download path using a UUID
        temp_dir = os.path.join("/tmp", "vectorize", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "prediction.geojson")
    polygonize(
        prediction_path,
        output_path,
        remove_inputs=False,
        simplify_tolerance=tolerance,
        merging_distance_threshold=merging_distance_threshold,
        area_threshold=area_threshold,
    )
    return output_path
