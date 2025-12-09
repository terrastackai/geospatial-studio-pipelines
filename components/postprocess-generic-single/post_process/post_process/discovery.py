# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pathlib
from .registry import POST_PROCESS_REGISTRY
from .logging import logger

def load_fs_plugins(directory: str = "post_process/post_process/generic"):
    for pyfile in pathlib.Path(directory).glob("*.py"):
        spec = importlib.util.spec_from_file_location(pyfile.stem, str(pyfile))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for {pyfile}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # this triggers @register_step decorators

        logger.info(f"Loaded post-process plugin from {pyfile} as module {module.__name__}")






