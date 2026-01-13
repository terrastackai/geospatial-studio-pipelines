# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Any, List

from .logging import logger


# Global registry of post-process functions
POST_PROCESS_REGISTRY: Dict[str, Callable] = {}

def register_step(name: str):
    """Decorator to register a post-process function."""
    def _decorator(func: Callable):
        if name in POST_PROCESS_REGISTRY:
            raise ValueError(f"Duplicate post-process step name: {name}")
        POST_PROCESS_REGISTRY[name] = func
        return func
    return _decorator

def run_chain(img, steps_config: List[Dict[str, Any]]):
    """
    Function to run the registered post_processed functions
    steps_config example:
    [
      {"name": "cloud_mask", "params": {"threshold": 0.2}},
      {"name": "contrast_stretch", "params": {"percentiles": [2, 98]}}
    ]
    """
    print(f"Requested steps: {steps_config}")    

    outputs = {}
    out = img
    for item in steps_config:
        name = item["name"]
        params = item.get("params", {})

        if name not in POST_PROCESS_REGISTRY:
            raise ValueError(f"Post-process step '{name}' not found.")
        
        print(f"Post-process step '{name}'  found.")

        step_fn = POST_PROCESS_REGISTRY[name]
        out = step_fn(out, **params)

        outputs[name] = out
    return outputs
