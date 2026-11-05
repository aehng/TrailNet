"""Export predicted trails to GPX, KML, or GeoJSON.

GPX is the format MTB Project and Trailforks accept for trail submissions
(one track per predicted trail). KML is for Google Earth and GIS tools.
GeoJSON is a clean re-export of the predicted trails with name, area, and
source properties. Note: MTB Project and Trailforks are curated platforms,
so an exported file is a submission for their review process, not an
automatic listing.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd

from trailnet.config import resolve_path

FORMATS = ("gpx", "kml", "geojson")


def _track_parts(gdf):
    """Yield (trail_id, [(lon, lat), ...]) for every line part in the frame."""
    for idx, row in gdf.iterrows():
        geom = row.geometry
        trail_id = str(row["trail_id"]) if "trail_id" in row and row["trail_id"] is not None else str(idx)
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            yield trail_id, list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            for part_i, part in enumerate(geom.geoms):
                yield f"{trail_id}_{part_i}", list(part.coords)


def _load_trails(geojson_path):
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def export_gpx(gdf, output_path, area_name=None):
    gpx = ET.Element("gpx", {
        "version": "1.1",
        "creator": "TrailNet",
        "xmlns": "http://www.topografix.com/GPX/1/1",
    })
    for trail_id, coords in _track_parts(gdf):
        trk = ET.SubElement(gpx, "trk")
        name = f"{area_name}_trail_{trail_id}" if area_name else f"trail_{trail_id}"
        ET.SubElement(trk, "name").text = name
        seg = ET.SubElement(trk, "trkseg")
        for lon, lat in coords:
            ET.SubElement(seg, "trkpt", {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(gpx)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def export_kml(gdf, output_path, area_name=None):
    kml = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = f"TrailNet — {area_name}" if area_name else "TrailNet"
    for trail_id, coords in _track_parts(gdf):
        place = ET.SubElement(doc, "Placemark")
        name = f"{area_name}_trail_{trail_id}" if area_name else f"trail_{trail_id}"
        ET.SubElement(place, "name").text = name
        line = ET.SubElement(place, "LineString")
        ET.SubElement(line, "coordinates").text = " ".join(
            f"{lon:.7f},{lat:.7f},0" for lon, lat in coords
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def export_geojson(gdf, output_path, area_name=None):
    out = gdf.copy()
    if "trail_id" in out:
        out["name"] = [f"{area_name}_trail_{row}" if area_name else f"trail_{row}"
                       for row in out["trail_id"]]
    else:
        out["name"] = [f"trail_{i}" for i in range(len(out))]
    out["area"] = area_name
    out["source"] = "TrailNet prediction"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(output_path, driver="GeoJSON")
    return output_path


def export_trails(geojson_path, output_path, fmt="gpx", area_name=None):
    """Export a predicted-trails GeoJSON file to GPX, KML, or GeoJSON."""
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}. Choose from: {', '.join(FORMATS)}")
    gdf = _load_trails(geojson_path)
    if fmt == "gpx":
        return export_gpx(gdf, output_path, area_name)
    if fmt == "kml":
        return export_kml(gdf, output_path, area_name)
    return export_geojson(gdf, output_path, area_name)


def export_area(config, area, fmt="gpx"):
    """Export one area's predicted trails. Returns the output path."""
    vector_dir = resolve_path(config, config["paths"]["vector_dir"])
    trails = vector_dir / f"{area['name']}_trails.geojson"
    if not trails.exists():
        raise FileNotFoundError(f"No predicted trails file found: {trails}")
    output = (resolve_path(config, config["paths"]["output_dir"]) / "exports"
              / f"{area['name']}_trails.{fmt}")
    return export_trails(trails, output, fmt, area["name"])
