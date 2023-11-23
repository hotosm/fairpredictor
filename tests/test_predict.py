bbox = [-84.1334429383278, 9.953153171808898, -84.13033694028854, 9.954719779271468]
zoom_level = 19
from predictor import download

image_download_path = download(
    bbox,
    zoom_level=zoom_level,
    tms_url="bing",
    tile_size=256,
    download_path="/Users/kshitij/hotosm/fairpredictor/download/test",
)
print(image_download_path)
