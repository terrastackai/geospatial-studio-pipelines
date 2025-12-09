#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster/Vector → Regularized Building Polygons (and optional rasterized output)

- Raster input (binary mask): polygonize → auto orientation correction → Building-Regulariser → vector/raster output
- Vector input: auto orientation correction → Building-Regulariser → vector/raster output

Requires: rasterio, geopandas, shapely>=2, numpy, buildingregulariser

Example (raster → vector):
  python im2poly_regularize.py \
    --input data/mask.tif \
    --output out/buildings.gpkg \
    --output-type vector \
    --prob-threshold 0.5 \
    --min-area 1.0 \
    --br-kwargs '{"simplify_tolerance":0.8,"parallel_threshold":2.0,"allow_45_degree":true}'

Example (vector → raster, using template):
  python im2poly_regularize.py \
    --input data/buildings_raw.shp \
    --output out/buildings_clean.tif \
    --output-type raster \
    --raster-template data/template.tif \
    --br-kwargs '{"simplify_tolerance":0.7,"allow_any_angle":true,"parallel_threshold":3.0}'

"""

import argparse
import json
import math
import os
from typing import Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
from shapely.affinity import rotate as shp_rotate

import rasterio
from rasterio.features import shapes as rio_shapes, rasterize as rio_rasterize
from rasterio.transform import from_origin

try:
    from buildingregulariser import regularize_geodataframe
except Exception as e:
    raise ImportError("buildingregulariser is required. Install via conda-forge or pip.")

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable

import numpy as np
import geopandas as gpd
from shapely import affinity
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Building-Regulariser (https://github.com/DPIRD-DMA/Building-Regulariser)
from buildingregulariser import regularize_geodataframe
import pandas as pd

from post_process.post_process.registry import register_step
# -------------------------
# Helpers
# -------------------------

def infer_driver(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gpkg":
        return "GPKG"
    if ext in (".geojson", ".json"):
        return "GeoJSON"
    if ext == ".shp":
        return "ESRI Shapefile"
    if ext == ".gpkg":
        return "GPKG"
    return "GPKG"


def auto_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Return an EPSG string for a UTM CRS based on centroid of bounds (expects geographic input)."""
    # If already projected, just return current
    if gdf.crs and not gdf.crs.is_geographic:
        return gdf.crs.to_string()
    # Compute lon/lat center
    tmp = gdf
    if not (gdf.crs and gdf.crs.is_geographic):
        tmp = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = tmp.total_bounds
    lon = (minx + maxx) / 2.0
    lat = (miny + maxy) / 2.0
    zone = int(math.floor((lon + 180) / 6) + 1)
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return f"EPSG:{epsg}"


def make_geoms_valid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf.copy()
    g["geometry"] = g.geometry.apply(make_valid)
    # Flatten geometry collections to polygons
    def _flatten(geom):
        if geom.is_empty:
            return None
        if geom.geom_type == "GeometryCollection":
            polys = [h for h in geom.geoms if h.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                return None
            return unary_union(polys)
        return geom
    g["geometry"] = g.geometry.apply(_flatten)
    g = g[~g.geometry.is_empty & g.geometry.notnull()].copy()
    return g


# -------------------------
# Raster → polygons
# -------------------------

from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
import geopandas as gpd


def polygonize_building_mask(
    raster_path: str,
    mask_band: int = 1,
    building_value: int | float = 1,
    background_value: int | float = 0,
    min_area_m2: float = 0.0,
    keep_holes: bool = False,
    simplify_tolerance: Optional[float] = None,
    dst_crs: Optional[str] = None,
    save_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load a binary building mask raster and convert it to polygons in a GeoDataFrame.

    Parameters
    ----------
    raster_path : str
        Path to the raster mask (buildings = building_value; background = background_value).
    mask_band : int, optional
        1-based band index containing the mask. Default is 1.
    building_value : int|float, optional
        Pixel value that represents buildings. Default 1.
    background_value : int|float, optional
        Pixel value that represents background. Default 0.
    min_area_m2 : float, optional
        Remove polygons with area smaller than this (in CRS units squared; if CRS is projected in meters, it's m²).
        Default 0 (keep all).
    keep_holes : bool, optional
        If False, remove interior rings (holes) from polygons. Default False.
    simplify_tolerance : float, optional
        If provided (>0), simplify geometries with this tolerance in CRS units (useful to reduce stair-steps).
        A good starting value is 1–2× the raster pixel size. Default None (no simplification).
    dst_crs : str, optional
        Reproject output to this CRS (e.g., 'EPSG:4326'). Default None (stay in source CRS).
    save_path : str, optional
        If provided, save the result to this path (suffix determines format, e.g., .gpkg, .geojson, .shp).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with one polygon per building. Columns: ['value'] + geometry.
    """
    # --- 1) Read raster & metadata ---
    with rasterio.open(raster_path) as src:
        mask = src.read(mask_band)
        transform = src.transform
        src_crs = src.crs

    # Ensure binary (in case of soft/probability masks); treat any non-background as building
    # You can adjust the thresholding here if your mask is probabilistic.
    binary = (mask != background_value) & (mask == building_value if building_value != 1 else (mask > 0))

    # --- 2) Polygonize using rasterio.features.shapes ---
    # shapes() yields (geom_mapping, value) for contiguous regions with the same value.
    # We pass mask=binary to extract only building regions, and image=binary.astype(np.uint8)
    # so the 'value' in results is 1 for building areas.
    results = shapes(
        source=binary.astype(np.uint8),
        mask=binary.astype(np.uint8),
        transform=transform
    )

    geoms = []
    vals = []

    for geom_mapping, val in results:
        if not val:
            continue  # Only keep building regions (val==1)
        geom = shape(geom_mapping)

        # Optionally drop holes (interior rings) for cleaner footprints
        if not keep_holes:
            if isinstance(geom, Polygon):
                geom = Polygon(geom.exterior)
            elif isinstance(geom, MultiPolygon):
                geom = MultiPolygon([Polygon(p.exterior) for p in geom.geoms if not p.is_empty])

        # Fix invalids (self-intersections, etc.)
        geom = make_valid(geom)

        # Some make_valid results can be GeometryCollection; reduce to (Multi)Polygon
        if geom.is_empty:
            continue
        if geom.geom_type == "GeometryCollection":
            # keep only polygonal pieces
            polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                continue
            geom = unary_union(polys)

        # Optional simplification (after validity fix)
        if simplify_tolerance and simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
            if geom.is_empty:
                continue

        geoms.append(geom)
        vals.append(int(val))

    if not geoms:
        # Return empty GeoDataFrame with CRS set if possible
        gdf = gpd.GeoDataFrame({"value": []}, geometry=[], crs=src_crs)
        if dst_crs:
            gdf = gdf.to_crs(dst_crs)
        if save_path:
            gdf.to_file(save_path)
        return gdf

    gdf = gpd.GeoDataFrame({"value": vals}, geometry=geoms, crs=src_crs)

    return gdf


# -------------------------
# Orientation estimation/correction
# -------------------------

# ---------- Helpers: orientation, rotation, cleaning ----------

def _poly_orientation_deg(poly: Polygon | MultiPolygon) -> float:
    """
    Estimate polygon orientation in degrees (-90..90] using the minimum rotated rectangle.
    Returns the angle of the longest side relative to +X axis.
    """
    if poly.is_empty:
        return 0.0
    try:
        # For MultiPolygon, use largest part by area
        if isinstance(poly, MultiPolygon):
            parts = [p for p in poly.geoms if not p.is_empty]
            if not parts:
                return 0.0
            poly = max(parts, key=lambda p: p.area)

        mrr = poly.minimum_rotated_rectangle
        # mrr is a 5-point ring; take consecutive edge vectors
        coords = np.asarray(mrr.exterior.coords)
        edges = coords[1:] - coords[:-1]  # 4 edges
        lengths = np.hypot(edges[:, 0], edges[:, 1])
        i = int(np.argmax(lengths))
        dx, dy = edges[i]
        ang = np.degrees(np.arctan2(dy, dx))  # (-180..180]
        # map to (-90, 90]: flipping 180 keeps axis equivalence
        if ang <= -90:
            ang += 180
        elif ang > 90:
            ang -= 180
        return float(ang)
    except Exception:
        return 0.0


def _area_weighted_dominant_angle(geoms: Iterable, weights: Optional[np.ndarray] = None) -> float:
    """
    Compute an area-weighted median orientation over a set of polygons.
    Returns angle in degrees (-90..90].
    """
    angles = []
    w = []
    for g in geoms:
        if g.is_empty:
            continue
        a = g.area
        if a <= 0:
            continue
        angles.append(_poly_orientation_deg(g))
        w.append(a)
    if not angles:
        return 0.0
    angles = np.asarray(angles, dtype=float)
    w = np.asarray(w, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    # weighted median
    order = np.argsort(angles)
    angles_sorted = angles[order]
    w_sorted = w[order]
    csum = np.cumsum(w_sorted) / w_sorted.sum()
    idx = np.searchsorted(csum, 0.5)
    return float(angles_sorted[min(idx, len(angles_sorted) - 1)])


def _rotate_gdf(gdf: gpd.GeoDataFrame, angle_deg: float, origin: Tuple[float, float] = (0.0, 0.0)) -> gpd.GeoDataFrame:
    """Rotate all geometries by angle_deg around a fixed origin (default CRS origin)."""
    return gdf.assign(geometry=gdf.geometry.apply(lambda g: affinity.rotate(g, angle_deg, origin=origin)))


def _clean_geometries(gdf: gpd.GeoDataFrame, min_area: Optional[float] = None) -> gpd.GeoDataFrame:
    """
    Fix validity and optionally drop tiny slivers by area (units = CRS units^2).
    """
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    if min_area is not None and min_area > 0:
        gdf = gdf[gdf.geometry.area >= float(min_area)]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def _bin_angles(angles_deg: np.ndarray, bin_size_deg: float) -> np.ndarray:
    """
    Bin angles (-90..90] by nearest bin center with wrap-around.
    """
    # Shift to [0..180), bin, then shift back if needed just for grouping
    shifted = (angles_deg + 90.0) % 180.0
    bins = np.round(shifted / bin_size_deg).astype(int)
    return bins


# ---------- Main function ----------

@dataclass
class RegularizeParams:
    """
    Parameters for buildingregulariser.regularize_geodataframe (supported knobs only).
    """
    simplify_tolerance: float = 0.8          # meters (choose 2–3x image pixel size)
    parallel_threshold: float = 2.0          # larger = less aggressive snapping
    allow_45_degree: bool = True             # diagonals common? keep True
    diagonal_threshold_reduction: float = 0  # optional: loosen diagonal snapping (degrees)
    allow_circles: bool = False              # near-circular detection
    circle_threshold: float = 0.9            # if allow_circles=True

def regularize_with_orientation_correction(
    gdf: gpd.GeoDataFrame,
    *,
    mode: str = "global",                # "global" or "local"
    target_crs: str = "EPSG:3857",      # metric CRS for meter-based tolerances
    min_area_m2: Optional[float] = None,
    angle_bin_size_deg: float = 10.0,   # used when mode="local"
    params: RegularizeParams = RegularizeParams(),
    return_intermediate: bool = False
) -> Dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame:
    """
    Regularize building polygons while correcting for orientation, using Building-Regulariser.

    Steps:
      1) Reproject to metric CRS (meters) and clean invalids / tiny slivers.
      2) Orientation correction:
         - mode="global": estimate dominant angle, rotate all polygons by -angle.
         - mode="local" : estimate per-polygon angle, bin polygons by angle (angle_bin_size_deg),
                          rotate each bin by its negative dominant angle.
      3) Run buildingregulariser.regularize_geodataframe with supported parameters.
      4) Rotate results back to original orientation and CRS.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input polygons (preferably building footprints).
    mode : {"global","local"}
        "global" -> single AOI-wide rotation; "local" -> per-angle-bin rotation.
    target_crs : str
        Metric CRS to use during regularization (e.g., local UTM or EPSG:3857).
    min_area_m2 : float or None
        Drop polygons with area below this threshold (in m²) *before* regularization.
    angle_bin_size_deg : float
        Angular bin size for "local" mode (e.g., 10°).
    params : RegularizeParams
        Regularization parameters (meters & degrees as noted).
    return_intermediate : bool
        If True, returns dict with intermediates; else returns final GeoDataFrame.

    Returns
    -------
    GeoDataFrame or dict of GeoDataFrames
    """
    if gdf.empty:
        return gdf.copy()

    if gdf.crs is None:
        warnings.warn("Input GeoDataFrame has no CRS. Assuming EPSG:4326; set .crs explicitly if different.")
        gdf = gdf.set_crs("EPSG:4326")

    src_crs = gdf.crs
    # -- Project to metric CRS for meter-based tolerances/areas
    gdf_m = gdf.to_crs(target_crs)
    gdf_m = _clean_geometries(gdf_m, min_area=min_area_m2)

    intermediates = {}

    if mode not in {"global", "local"}:
        raise ValueError("mode must be 'global' or 'local'.")

    if mode == "global":
        # 1) Estimate AOI dominant orientation (area-weighted)
        dom_ang = _area_weighted_dominant_angle(gdf_m.geometry)
        intermediates["dominant_angle_deg"] = dom_ang

        # 2) Rotate to align with axes
        gdf_rot = _rotate_gdf(gdf_m, -dom_ang, origin=(0.0, 0.0))

        # 3) Regularize (supported params only)
        gdf_reg_rot = regularize_geodataframe(
            gdf_rot,
            simplify_tolerance=params.simplify_tolerance,
            parallel_threshold=params.parallel_threshold,
            allow_45_degree=params.allow_45_degree,
            diagonal_threshold_reduction=params.diagonal_threshold_reduction,
            allow_circles=params.allow_circles,
            circle_threshold=params.circle_threshold,
        )

        # Optional second clean (safety)
        gdf_reg_rot = _clean_geometries(gdf_reg_rot)

        # 4) Rotate back
        gdf_reg_m = _rotate_gdf(gdf_reg_rot, +dom_ang, origin=(0.0, 0.0))

    else:  # mode == "local"
        # Compute per-polygon angles
        angles = np.array([_poly_orientation_deg(geom) for geom in gdf_m.geometry], dtype=float)
        bins = _bin_angles(angles, angle_bin_size_deg)
        gdf_m = gdf_m.assign(_angle=angles, _bin=bins)

        parts = []
        bin_stats = {}

        for b in np.unique(bins):
            sub = gdf_m[gdf_m._bin == b].copy()
            if sub.empty:
                continue
            # dominant angle of this bin (area-weighted)
            dom_ang_b = _area_weighted_dominant_angle(sub.geometry)
            bin_stats[int(b)] = float(dom_ang_b)
            # rotate bin, regularize, rotate back
            sub_rot = _rotate_gdf(sub, -dom_ang_b, origin=(0.0, 0.0))
            sub_reg_rot = regularize_geodataframe(
                sub_rot,
                simplify_tolerance=params.simplify_tolerance,
                parallel_threshold=params.parallel_threshold,
                allow_45_degree=params.allow_45_degree,
                diagonal_threshold_reduction=params.diagonal_threshold_reduction,
                allow_circles=params.allow_circles,
                circle_threshold=params.circle_threshold,
            )
            sub_reg_rot = _clean_geometries(sub_reg_rot)
            sub_reg = _rotate_gdf(sub_reg_rot, +dom_ang_b, origin=(0.0, 0.0))
            # keep original attributes if needed (here we just keep geometry/value-like columns)
            parts.append(sub_reg.drop(columns=[c for c in sub_reg.columns if c in ("_angle", "_bin") and c in sub_reg.columns], errors="ignore"))

        gdf_reg_m = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame(columns=gdf_m.columns, geometry=[]), crs=gdf_m.crs)
        intermediates["bin_stats_deg"] = bin_stats

    # Final clean and back to source CRS
    gdf_reg_m = _clean_geometries(gdf_reg_m)
    gdf_reg = gdf_reg_m.to_crs(src_crs)

    if return_intermediate:
        intermediates["input_metric"] = gdf_m
        intermediates["output_metric"] = gdf_reg_m
        intermediates["output"] = gdf_reg
        return intermediates
    return gdf_reg

@register_step("im2poly_regularize")
def im2poly_entrypoint(func, img_path: str, out_path: str, params: RegularizeParams, **_):
    """Entrypoint function to run the im2poly regularization function

    Parameters
    ----------
    img_path : str
        Path to the img to regularize
    out_path : str
        Output path to save outputs
    params : RegularizeParams
        Parameters for the regularize function
    """
    gdf = polygonize_building_mask(
        raster_path=img_path, save_path=img_path.replace(".tif", ".gpkg")
    )
    reg = regularize_with_orientation_correction(
        gdf,
        mode="global",
        target_crs="EPSG:3857",
        min_area_m2=1.0,  # drop speckles
        params=RegularizeParams(**params),
        return_intermediate=True,
    )

    clean = reg["output"]
    clean.to_file(out_path)

    return {
        "processed_file": clean,
        "processed_file_path": out_path,
    }
