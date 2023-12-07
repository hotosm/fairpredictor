import concurrent.futures
import math
import os
import re
from glob import glob
from typing import List, Tuple

import geopandas
import numpy as np
import requests
from PIL import Image
from shapely.geometry import box

try:
    from tensorflow import keras
except ImportError:
    pass

IMAGE_SIZE = 256


def get_start_end_download_coords(bbox_coords, zm_level, tile_size):
    # start point where we will start downloading the tiles

    start_point_lng = bbox_coords[0]  # getting the starting lat lng
    start_point_lat = bbox_coords[1]

    # end point where we should stop downloading the tile
    end_point_lng = bbox_coords[2]  # getting the ending lat lng
    end_point_lat = bbox_coords[3]

    # Note :  lat=y-axis, lng=x-axis
    # getting tile coordinate for first point of bbox
    start_x, start_y = latlng2tile(
        zoom=zm_level,
        lat=start_point_lat,
        lng=start_point_lng,
        tile_size=tile_size,
    )
    start = [start_x, start_y]

    # getting tile coordinate for last point of bbox
    end_x, end_y = latlng2tile(
        zoom=zm_level,
        lat=end_point_lat,
        lng=end_point_lng,
        tile_size=tile_size,
    )
    end = [end_x, end_y]
    return start, end


def download_image(url, base_path, source_name):
    response = requests.get(url)

    image = response.content

    pattern = r"/(\d+)/(\d+)/(\d+)(?:\.\w+)?"
    match = re.search(pattern, url)
    # filename = z-x-y
    filename = f"{base_path}/{source_name}-{match.group(2)}-{match.group(3)}-{match.group(1)}.png"

    with open(filename, "wb") as f:
        f.write(image)


def convert2worldcd(lat, lng, tile_size):
    """
    World coordinates  are measured from the Mercator projection's origin
    (the northwest corner of the map at 180 degrees longitude and
    approximately 85 degrees latitude) and increase in the x direction
    towards the east (right) and increase in the y direction towards the south
    (down).Because the basic Mercator  tile is 256 x 256 pixels, the usable
    world coordinate space is {0-256}, {0-256}
    """
    siny = math.sin((lat * math.pi) / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    world_x = tile_size * (0.5 + (lng / 360))
    world_y = tile_size * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    # print("world coordinate space is %s, %s",world_x,world_y)
    return world_x, world_y


def latlng2tile(zoom, lat, lng, tile_size):
    """By dividing the pixel coordinates by the tile size and taking the
    integer parts of the result, you produce as a by-product the tile
    coordinate at the current zoom level."""
    zoom_byte = 1 << zoom  # converting zoom level to pixel bytes
    # print(zoom_byte)
    w_x, w_y = convert2worldcd(lat, lng, tile_size)

    t_x = math.floor((w_x * zoom_byte) / tile_size)
    t_y = math.floor((w_y * zoom_byte) / tile_size)
    return t_x, t_y


def tile_xy_to_quad_key(tile_x, tile_y, level_of_detail):
    quad_key = []
    for i in range(level_of_detail, 0, -1):
        digit = "0"
        mask = 1 << (i - 1)
        if (tile_x & mask) != 0:
            digit = chr(ord(digit) + 1)
        if (tile_y & mask) != 0:
            digit = chr(ord(digit) + 1)
            digit = chr(ord(digit) + 1)
        quad_key.append(digit)

    return "".join(quad_key)


def download_imagery(start: list, end: list, zm_level, base_path, source="maxar"):
    """Downloads imagery from start to end tile coordinate system

    Args:
        start (list):[tile_x,tile_y]
        end (list): [tile_x,tile_y],
        source (string): it should be eithre url string or maxar value
        zm_level : Zoom level
    """

    begin_x = start[0]  # this will be the beginning of the download loop for x
    begin_y = start[1]  # this will be the beginning of the download loop for x
    stop_x = end[0]  # this will be the end of the download loop for x
    stop_y = end[1]  # this will be the end of the download loop for x

    print(f"Download starting from {start} to {end} using source {source} - {zm_level}")

    start_x = begin_x  # starting loop from beginning
    start_y = begin_y  # starting y loop from beginnig
    source_name = "OAM"  # default
    download_urls = []
    while start_x <= stop_x:  # download  x section while keeping y as c
        start_y = begin_y
        while start_y >= stop_y:  # download  y section while keeping x as c
            download_path = [start_x, start_y]
            if source == "maxar":
                try:
                    connect_id = os.environ.get("MAXAR_CONNECT_ID")
                except Exception as ex:
                    raise ex
                source_name = source
                download_url = f"https://services.digitalglobe.com/earthservice/tmsaccess/tms/1.0.0/DigitalGlobe:ImageryTileService@EPSG:3857@jpg/{zm_level}/{download_path[0]}/{download_path[1]}.jpg?connectId={connect_id}&flipy=true"
            elif source == "bing":
                download_url = f"https://ecn.t2.tiles.virtualearth.net/tiles/a{tile_xy_to_quad_key(download_path[0],download_path[1],zm_level)}.jpeg?g=14037&pr=odbl&n=z"
                print(download_url)
            else:
                # source should be url as string , like this :  https://tiles.openaerialmap.org/62dbd947d8499800053796ec/0/62dbd947d8499800053796ed/{z}/{x}/{y}
                if "{-y}" in source:
                    ## negative TMS
                    source_value = source.replace("{-y}", "{y}")
                    # conversion from normal tms
                    y_value = int((2**zm_level) - download_path[1] - 1)

                else:
                    # If it doesn't, use the positive y-coordinate
                    y_value = download_path[1]
                    source_value = source
                download_url = source_value.format(
                    x=download_path[0], y=y_value, z=zm_level
                )

            download_urls.append(download_url)

            start_y = start_y - 1  # decrease the y

        start_x = start_x + 1  # increase the x

    # Use the ThreadPoolExecutor to download the images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for url in download_urls:
            executor.submit(download_image, url, base_path, source_name)


def get_bounding_box(filename: str) -> Tuple[float, float, float, float]:
    """Get the EPSG:3857 coordinates of bounding box for the OAM image.

    This function gives the coordinates of lower left and upper right
    corners of the OAM image. We will use the bounding box to georeference
    the image and for clipping and rasterizing the corresponding labels.

    Returns:
        A tuple, (x_min, y_min, x_max, y_max), with coordinates in meters.
    """
    filename = re.sub(r"\.(png|jpeg)$", "", filename)
    _, *tile_info = re.split("-", filename)
    x_tile, y_tile, zoom = map(int, tile_info)

    # Lower left and upper right corners in degrees
    x_min, y_min = num2deg(x_tile, y_tile + 1, zoom)
    x_max, y_max = num2deg(x_tile + 1, y_tile, zoom)

    # Create a GeoDataFrame containing a polygon for bounding box
    box_4326 = box(x_min, y_min, x_max, y_max)
    gdf_4326 = geopandas.GeoDataFrame({"geometry": [box_4326]}, crs="EPSG:4326")

    # Reproject to EPSG:3857
    gdf_3857 = gdf_4326.to_crs("EPSG:3857")

    # Bounding box in EPSG:3857 as a tuple (x_min, y_min, x_max, y_max)
    box_3857 = gdf_3857.iloc[0, 0].bounds

    return box_3857


def num2deg(x_tile: int, y_tile: int, zoom: int) -> Tuple[float, float]:
    """Convert tile numbers to EPSG:4326 coordinates.

    Convert tile numbers to the WGS84 longitude/latitude coordinates
    (in degrees) of the upper left corner of the tile.

    Args:
        x_tile: Tile X coordinate
        y_tile: Tile Y coordinate
        zoom: Level of detail

    Returns:
        A tuple (longitude, latitude) in degrees.
    """
    n = 2.0**zoom
    lon_deg = x_tile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
    lat_deg = math.degrees(lat_rad)

    return lon_deg, lat_deg


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
    files = glob(pattern)
    for file in files:
        os.remove(file)


def save_mask(mask: np.ndarray, filename: str) -> None:
    """Save the mask array to the specified location."""
    reshaped_mask = mask.reshape((IMAGE_SIZE, IMAGE_SIZE)) * 255
    result = Image.fromarray(reshaped_mask.astype(np.uint8))
    result.save(filename)
