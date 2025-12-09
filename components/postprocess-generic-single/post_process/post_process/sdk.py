# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import pathlib
import shutil
import ast
from typing import List, Dict, Union, Tuple
import os
import urllib.request
import urllib.error
import ssl
import re
from urllib.parse import urlparse, unquote

from .logging import logger


STEPS_DIR = pathlib.Path(__file__).parent / "generic"


def add_step(
    script_path: str,
    fail_on_duplicate_content: bool = True,
    allow_version_suffix: bool = False,
):
    """Function to add a script to the folder

    Parameters
    ----------
    script_path : str
        path to script
    fail_on_duplicate_content : bool, optional
        _description_, by default True
    allow_version_suffix : bool, optional
        Allow duplicate files, by default False

    Raises
    ------
    FileNotFoundError
        Script file not found.
    ValueError
        Invalid script type provided.
    FileExistsError
        Script file already exists.
    ValueError
        No @register_step functions in the script.
    """

    src = pathlib.Path(script_path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Script not found: {src}")
    if src.suffix != ".py":
        raise ValueError("Invalid file type: expected a .py script")

    dest = STEPS_DIR / src.name
    if dest.exists():
        if not allow_version_suffix:
            raise FileExistsError(
                f"Destination file already exists: {dest.name}. "
                "Set allow_version_suffix=True to auto-version, or rename your file."
            )

    # Parse and validate decorators before copying
    steps_in_file = _parse_registered_steps(src)
    if not steps_in_file:
        raise ValueError(
            f"No @register_step functions found in {src}. "
            "Ensure your functions are decorated, e.g., @register_step(name='masking')."
        )

    # Ensure steps directory exists
    STEPS_DIR.mkdir(exist_ok=True)

    # Copy script into steps directory
    dest = STEPS_DIR / src.name
    shutil.copy(src, dest)

    # # Dynamically load the module
    # load_fs_plugins(STEPS_DIR)
    # spec = importlib.util.spec_from_file_location(dest.stem, dest)
    # module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(module)

    logger.info(
        f"✅ Step '{dest.stem}' added to {STEPS_DIR} successfully.\n Run `load_fs_plugins` to register the function"
    )


def _parse_registered_steps(source_path: pathlib.Path) -> List[str]:
    """
    Statically parse the file to find functions decorated with @register_step(...).
    We intentionally avoid import execution here (security + correctness).
    """
    text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source_path))
    steps = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.decorator_list:
                continue
            for dec in node.decorator_list:
                # Match @register_step or @register_step(...)
                if isinstance(dec, ast.Name) and dec.id == "register_step":
                    # Name can be provided via function name fallback
                    steps.append(node.name)
                elif isinstance(dec, ast.Call) and (
                    (isinstance(dec.func, ast.Name) and dec.func.id == "register_step")
                    or (
                        isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == "register_step"
                    )
                ):
                    # look for keyword arg name="..."
                    step_name = None
                    for kw in dec.keywords or []:
                        if kw.arg == "name":
                            if isinstance(kw.value, ast.Str):
                                step_name = kw.value.s
                            elif isinstance(kw.value, ast.Constant) and isinstance(
                                kw.value.value, str
                            ):
                                step_name = kw.value.value
                            break
                    steps.append(step_name or node.name)
    return sorted(set(steps))


def _safe_filename_from_url(url: str) -> str:
    """Generate a safe filename from a URL."""

    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename:
        filename = "plugin.py"
    filename = unquote(filename)
    # Remove unsafe characters
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return filename


def download_plugins_to_local(
    plugins_list: Union[List[Union[str, Dict]], None],
    tmp_plugins_dir: str = "/tmp/post_process/",
    verify_ssl: bool = True,
    timeout: int = 30,
    **kwargs,
) -> Tuple[List[Dict], List[Dict]]:
    """Download plugin python files to local tmp directory.

    plugins_list elements may be:
        - a string URL to a .py file
        - a dict with keys: "url" (required), optional "filename" or "name" used to save file,
        optional "module_name" used as import hint.

    Downloads each plugin into plugins_dir (created if missing).

    Returns a Tuple of list of dicts describing successful and failed attempts.
    """
    results = []
    saved_plugins = []
    if not plugins_list:
        logger.info("No post-processing scripts provided")
        return results

    os.makedirs(tmp_plugins_dir, exist_ok=True)

    if plugins_list is None:
        logger.info("No post-processing scripts provided")
        return results

    for entry in plugins_list:
        try:
            if isinstance(entry, str):
                url = entry
                filename = None
                module_name_hint = None
            elif isinstance(entry, dict):
                url = entry.get("url")
                filename = entry.get("filename") or entry.get("name")
            else:
                logger.warning(f"⚠️ Unsupported plugin entry type: {type(entry)}")
                results.append(
                    {"entry": entry, "status": "skipped", "reason": "bad_entry_type"}
                )
                continue

            if not url:
                logger.warning(f"⚠️ Plugin entry missing url: {entry}")
                results.append(
                    {"entry": entry, "status": "skipped", "reason": "missing_url"}
                )
                continue
            if not filename:
                filename = _safe_filename_from_url(url)

            # Allow any file type to be downloaded; we will attempt to import only .py files later.
            # Download the file
            filepath = os.path.join(tmp_plugins_dir, filename)
            logger.info(f" ⬇️  Downloading plugin from {url} to {filepath}")

            def _do_request(context):
                with urllib.request.urlopen(
                    url, timeout=timeout, context=context
                ) as resp:
                    if getattr(resp, "status", None) and resp.status >= 400:
                        raise urllib.error.HTTPError(
                            url, resp.status, resp.reason, resp.headers, None
                        )
                    return resp.read()

            try:
                # First attempt: default SSL verification
                ctx = None if verify_ssl else ssl._create_unverified_context()
                content = _do_request(ctx)
            except urllib.error.URLError as e:
                # Catch certificate verification failures and retry with unverified context
                err_str = str(e)
                if (
                    verify_ssl
                    and "CERTIFICATE_VERIFY_FAILED" in err_str
                    or (hasattr(e, "reason") and isinstance(e.reason, ssl.SSLError))
                ):
                    logger.exception(
                        f"❌ Failed to download plugin {filepath} from {url} due to SSL verification failures"
                    )
                    results.append(
                        {f"entry": entry, "status": "download_failed", "reason": str(e)}
                    )
                    continue
                else:
                    logger.exception(
                        f"❌ Failed to download plugin {filepath} from {url}"
                    )
                    results.append(
                        {"entry": entry, "status": "download_failed", "reason": str(e)}
                    )
                    continue
            except Exception as e:
                logger.exception(f"❌ Failed to download plugin from {url}")
                results.append(
                    {"entry": entry, "status": "download_failed", "reason": str(e)}
                )
                continue

            # Write to disk
            try:
                with open(filepath, "wb") as fh:
                    fh.write(content)
                # Add the step to the registry directory.
                # Don't load it here, it will be loaded when running post-processing in main file
                try:
                    add_step(
                        script_path=filepath,
                        allow_version_suffix=False,
                        **kwargs
                    )
                    results.append(
                        {
                            "entry": entry,
                            "status": "writen_to_disk",
                            "filename": filename,
                            "path": filepath,
                        }
                    )
                    saved_plugins.append(
                        {
                            "entry": entry,
                            "status": "writen_to_disk",
                            "filename": filename,
                            "path": filepath,
                        }
                    )
                    logger.info(f"✅ Plugin script {filepath} added successfully")
                except Exception as e:
                    logger.exception(f"❌ Failed to add step from script {filepath}")
                    results.append(
                        {
                            "entry": entry,
                            "status": "add_step_failed",
                            "reason": str(e),
                            "path": filepath,
                        }
                    )
                    continue
            except Exception as e:
                logger.exception(f"❌ Failed to write script to {filepath}")
                results.append(
                    {"entry": entry, "status": "write_failed", "reason": str(e)}
                )
                continue

            # Only attempt import for .py files
            if not filepath.endswith(".py"):
                logger.info(
                    f"✅ Plugin saved but not a .py file, skipping import: {filepath}",
                )
                results.append(
                    {"entry": entry, "status": "saved_not_py", "path": filepath}
                )
                continue

        except Exception as e:
            logger.exception(f"❌ Unexpected error processing plugin entry {entry}")
            results.append({"entry": entry, "status": "error", "reason": str(e)})

    return results, saved_plugins
