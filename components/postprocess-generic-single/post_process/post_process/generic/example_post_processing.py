# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import rasterio
import numpy as np
from post_process.post_process.registry import register_step

@register_step("example_post_processing")
def example_post_processing(func, img_path:str, nodata_value: float = 0, **_):
    print("Running Ex. post_processing step")

    with rasterio.open(img_path) as src:
        mask = src.read(1)
        nodata = src.nodata
        src_crs = src.crs
        # print(f"Unique: {np.unique(mask)}\nCRS: {src_crs}\nNodata: {nodata}")

        # At the moment same input_path returned as output_path
        return {
            "processed_file": img_path,
            "processed_file_path": img_path,
        }
