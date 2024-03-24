#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
# Source : https://github.com/devglobalpartners/ramp-code/blob/main/scripts/get_labels_from_masks.py
#################################################################


import os
import warnings
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from ramp.utils.geo_utils import (
    fill_rings,
    get_polygon_indices_to_merge,
    gpd_add_area_perim,
)
from ramp.utils.mask_to_vec_utils import binary_mask_from_multichannel_mask
from ramp.utils.misc_ramp_utils import dir_path
from ramp.utils.multimask_utils import binary_mask_to_geodataframe, buffer_df_geoms
from ramp.utils.ramp_exceptions import GdalReadError
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm


def convert_multimasks_to_geojson(
    input_dir,
    out_json,
    bdry_width_pixels=2,
    temp_mask_dir=None,
    out_crs="EPSG:4326",
    verbose=False,
):
    """
    Create fused geojson polygon outputs from a directory of multichannel masks.
    Use with truth masks to get fused truth building data, and with predicted masks to
    get fused building predictions.

    Parameters:
    - input_dir: Path to directory containing input multichannel masks.
    - out_json: Path to output label file (geojson)
    - bdry_width_pixels: The width of the building boundaries in the multichannel masks used in training. Default: 2.
    - temp_mask_dir: Temporary directory for writing reprojected masks. Optional.
    - out_crs: EPSG code for output label file CRS. Optional; default is "EPSG:4326".
    - verbose: Set to True for debug output. Optional; default is False.
    """

    mask_dir_path = Path(input_dir)
    assert (
        mask_dir_path.is_dir()
    ), f"Mask directory {str(mask_dir_path)} is not readable"
    mpaths = list(mask_dir_path.glob("**/*.tif"))
    num_files = len(mpaths)
    if verbose:
        print(f"Number of mask files to process: {num_files}")
    if num_files < 1:
        if verbose:
            print("No files to process: terminating")
        return

    output_filepath = Path(out_json)
    output_filepath.parent.mkdir(exist_ok=True, parents=True)

    buffer_pixels = bdry_width_pixels * 2 + 1

    transform = None
    building_dfs = []

    for mask_path in tqdm(mpaths):

        image_id = mask_path.name

        with rio.open(str(mask_path)) as mask_src:

            kwargs = mask_src.meta.copy()

            transform = mask_src.profile["transform"]
            mask_array = mask_src.read(1)

        # if (sparse) multichannel mask, convert to a binary
        binmask = binary_mask_from_multichannel_mask(mask_array)
        building_df = binary_mask_to_geodataframe(
            binmask,
            transform=transform,
            df_crs=kwargs["crs"],
            do_transform=True,  # we want output in degrees, not pixels
            min_area=0,  # set this to 0 to avoid problems with coordina
        )

        num_bldgs = len(building_df)
        if num_bldgs != 0:
            file_list = [image_id] * num_bldgs
            building_df["image_id"] = file_list
            building_dfs.append(building_df)

    num_dfs = len(building_dfs)
    if num_dfs < 1:
        if verbose:
            print("No buildings were extracted: terminating")
        return

    full_geodf = pd.concat(building_dfs, axis=0, ignore_index=True)

    full_geodf.to_file(str(output_filepath), driver="GeoJSON")
    return output_filepath

    # full_geodf["polyid"] = range(len(full_geodf))

    # ### polygon post processing

    # buff_geodf = buffer_df_geoms(full_geodf, buffer_pixels, affine_obj=transform)

    # df_join = full_geodf.sjoin(buff_geodf, how="left")
    # df_filtered = df_join[df_join["image_id_left"] != df_join["image_id_right"]]

    # full_geodf = get_polygon_indices_to_merge(full_geodf, df_filtered)

    # buff_geodf["merge_class"] = full_geodf["merge_class"]
    # merged_buffered_geodf = buff_geodf.dissolve(by="merge_class")
    # merged_geodf = buffer_df_geoms(
    #     merged_buffered_geodf, -buffer_pixels, affine_obj=transform
    # )

    # merged_geodf["geometry"] = merged_geodf.geometry.apply(lambda p: fill_rings(p))
    # merged_geodf = buffer_df_geoms(
    #     merged_geodf, bdry_width_pixels, affine_obj=transform
    # )
    # merged_geodf = gpd_add_area_perim(merged_geodf)

    # if len(merged_geodf) > 0:
    #     if out_crs != "EPSG:4326":
    #         merged_geodf = merged_geodf.to_crs(out_crs)
    #     if verbose:
    #         print(f"Writing label polygons to file: {str(output_filepath)}")
    #     merged_geodf.to_file(str(output_filepath), driver="GeoJSON")
    # else:
    #     if verbose:
    #         print("Output dataframe is empty: no file written")
    # return output_filepath
