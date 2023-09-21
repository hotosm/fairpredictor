import os
import uuid

from .utils import download_imagery, get_start_end_download_coords


def download(
    bbox,
    zoom_level,
    tms_url,
    tile_size=256,
    download_path=None,
):
    if download_path is None:
        # Generate a temporary download path using a UUID
        temp_dir = os.path.join("/tmp", "download", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        download_path = temp_dir

    start, end = get_start_end_download_coords(bbox, zoom_level, tile_size)
    download_imagery(
        start,
        end,
        zoom_level,
        base_path=download_path,
        source=tms_url,
    )
    return download_path
