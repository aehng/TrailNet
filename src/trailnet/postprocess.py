"""Probability raster to simplified trail-like linework and GeoJSON."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import xy
from shapely.geometry import LineString
from skimage.morphology import skeletonize

from trailnet.config import load_config, resolve_path
from trailnet.viz import build_map_area


def _skeleton_lines(skeleton, transform, confidence=None):
    points = {(int(r), int(c)) for r, c in np.argwhere(skeleton)}
    neighbors = {}
    for r, c in points:
        neighbors[(r, c)] = [(r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                             if (dr or dc) and (r+dr, c+dc) in points]
    visited = set()
    segments = []
    starts = [p for p in points if len(neighbors[p]) != 2]
    for start in starts:
        for nxt in neighbors[start]:
            edge = frozenset((start, nxt))
            if edge in visited:
                continue
            visited.add(edge); chain = [start, nxt]; previous, current = start, nxt
            while len(neighbors[current]) == 2:
                candidate = neighbors[current][0] if neighbors[current][1] == previous else neighbors[current][1]
                edge = frozenset((current, candidate))
                if edge in visited:
                    break
                visited.add(edge); chain.append(candidate); previous, current = current, candidate
            if len(chain) > 1:
                coords = [xy(transform, r, c, offset="center") for r, c in chain]
                line = LineString(coords)
                if line.length > 0:
                    score = float(np.mean([confidence[r, c] for r, c in chain])) if confidence is not None else None
                    segments.append((line, score))
    # Closed loops have no endpoint/junction nodes.
    for start in points:
        for nxt in neighbors[start]:
            edge = frozenset((start, nxt))
            if edge not in visited:
                chain = [start, nxt]; visited.add(edge); previous, current = start, nxt
                while current != start:
                    options = [p for p in neighbors[current] if p != previous]
                    if not options:
                        break
                    candidate = options[0]; edge = frozenset((current, candidate))
                    if edge in visited and candidate != start:
                        break
                    visited.add(edge); chain.append(candidate); previous, current = current, candidate
                if len(chain) > 2:
                    line = LineString([xy(transform, r, c, offset="center") for r, c in chain])
                    score = float(np.mean([confidence[r, c] for r, c in chain])) if confidence is not None else None
                    segments.append((line, score))
    return segments


def vectorize_probability(probability_path, output_path, threshold=0.5, simplify_m=1.0,
                          existing_trails=None, dedup_tolerance_m=3.0, area_name=None):
    with rasterio.open(probability_path) as ds:
        probability = ds.read(1)
        transform, crs = ds.transform, ds.crs
    if not crs or not crs.is_projected:
        raise ValueError("Probability raster must use a projected CRS with metric units")
    skeleton = skeletonize(probability >= threshold)
    line_scores = _skeleton_lines(skeleton, transform, probability)
    line_scores = [(line.simplify(simplify_m, preserve_topology=True), score)
                   for line, score in line_scores if line.length > simplify_m]
    if existing_trails is not None and Path(existing_trails).exists() and line_scores:
        existing = gpd.read_file(existing_trails).to_crs(crs)
        tree = existing.sindex
        kept = []
        for line, score in line_scores:
            candidates = list(tree.query(line.buffer(dedup_tolerance_m)))
            if not any(line.distance(existing.geometry.iloc[int(i)]) <= dedup_tolerance_m for i in candidates):
                kept.append((line, score))
        line_scores = kept
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    source = "TrailNet prediction"
    gdf = gpd.GeoDataFrame({
        "trail_id": list(range(len(line_scores))),
        "name": [f"{area_name or 'trail'}_{i + 1:04d}" for i in range(len(line_scores))],
        "area": [area_name] * len(line_scores),
        "source": [source] * len(line_scores),
        "confidence": [score for _, score in line_scores],
        "geometry": [line for line, _ in line_scores],
    }, geometry="geometry", crs=crs)
    # GeoJSON coordinates are emitted in longitude/latitude per RFC 7946.
    gdf.to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
    return output_path


def vectorize_area(config, area):
    probability = resolve_path(config, config["paths"]["prediction_dir"]) / f"{area['name']}_probability.tif"
    output = resolve_path(config, config["paths"]["vector_dir"]) / f"{area['name']}_trails.geojson"
    pp = config["postprocess"]
    written = vectorize_probability(probability, output, pp["threshold"], pp["simplify_m"],
                                    resolve_path(config, area["trails"]), pp["dedup_tolerance_m"],
                                    area_name=area["name"])
    build_map_area(config, area)
    return written
