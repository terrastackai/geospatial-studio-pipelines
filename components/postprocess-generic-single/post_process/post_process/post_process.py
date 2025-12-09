# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict, Any
from .registry import run_chain
from .discovery import load_fs_plugins

def post_process(img, steps_config: List[Dict[str, Any]], plugins_dir: str | None = None):
    """Function to post_process based on the steps_config provided.


    Parameters
    ----------
    img : Rioxarray image
        Rioxarray image read
    steps_config : List[Dict[str, Any]]
        List of dicts of post_processing steps. Ex.
           [
                {"name": "cloud_mask", "params": {"threshold": 0.2}},
                {"name": "contrast_stretch", "params": {"percentiles": [2, 98]}}
           ]
    plugins_dir : str | None, optional
        Path to dir with the post_process scripts registered, by default None

    Raises
    ------
    RuntimeError
        If no post_processing steps are provided in steps_config
    """
    # discover plugins at runtime
    if plugins_dir:
        load_fs_plugins(plugins_dir)

    # optional: assert registry not empty when config has steps
    if not steps_config:
        raise RuntimeError("No post-process steps registered.")

    # run sequential chain
    return run_chain(img=img,steps_config=steps_config)
