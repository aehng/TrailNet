"""Export predicted trails to GPX, KML, or GeoJSON.

GPX is the format MTB Project and Trailforks accept for trail submissions
(one track per predicted trail). KML is for Google Earth and GIS tools.
Exports preserve trail names, area, source, and confidence metadata. Note:
MTB Project and Trailforks are curated platforms,
so an exported file is a submission for their review process, not an
automatic listing.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd

from trailnet.config import resolve_path

FORMATS = ("gpx", "kml", "geojson")
TRAILNET_XMLNS = "https://trailnet.org/gpx/extensions/1.0"
ET.register_namespace("trailnet", TRAILNET_XMLNS)


def _metadata_value(row, key, fallback=None):
    value = row.get(key, fallback)
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    return value


def _track_parts(gdf):
    """Yield (row, part_id, coordinates) for every line part in the frame."""
    for idx, row in gdf.iterrows():
        geom = row.geometry
        trail_id = str(row["trail_id"]) if "trail_id" in row and row["trail_id"] is not None else str(idx)
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            yield row, trail_id, list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            for part_i, part in enumerate(geom.geoms):
                yield row, f"{trail_id}_{part_i}", list(part.coords)


def _load_trails(geojson_path):
    from trailnet.viz import _to_wgs84_or_raise
    gdf = gpd.read_file(geojson_path)
    return _to_wgs84_or_raise(gdf, source=str(geojson_path))


def export_gpx(gdf, output_path, area_name=None):
    gpx = ET.Element("gpx", {
        "version": "1.1",
        "creator": "TrailNet",
        "xmlns": "http://www.topografix.com/GPX/1/1",
    })
    for row, trail_id, coords in _track_parts(gdf):
        trk = ET.SubElement(gpx, "trk")
        name = str(_metadata_value(row, "name") or
                   (f"{area_name}_trail_{trail_id}" if area_name else f"trail_{trail_id}"))
        ET.SubElement(trk, "name").text = name
        metadata = {
            "area": _metadata_value(row, "area", area_name),
            "source": _metadata_value(row, "source", "TrailNet prediction"),
            "confidence": _metadata_value(row, "confidence"),
        }
        description = "; ".join(f"{key}={value}" for key, value in metadata.items() if value is not None)
        if description:
            ET.SubElement(trk, "desc").text = description
            extensions = ET.SubElement(trk, "extensions")
            for key, value in metadata.items():
                if value is not None:
                    ET.SubElement(extensions, f"{{{TRAILNET_XMLNS}}}{key}").text = str(value)
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
    for row, trail_id, coords in _track_parts(gdf):
        place = ET.SubElement(doc, "Placemark")
        name = str(_metadata_value(row, "name") or
                   (f"{area_name}_trail_{trail_id}" if area_name else f"trail_{trail_id}"))
        ET.SubElement(place, "name").text = name
        extended = ET.SubElement(place, "ExtendedData")
        metadata = {
            "area": _metadata_value(row, "area", area_name),
            "source": _metadata_value(row, "source", "TrailNet prediction"),
            "confidence": _metadata_value(row, "confidence"),
        }
        for key, value in metadata.items():
            if value is not None:
                data = ET.SubElement(extended, "Data", {"name": key})
                ET.SubElement(data, "value").text = str(value)
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
        fallback_names = [f"{area_name}_trail_{trail_id}" if area_name else f"trail_{trail_id}"
                          for trail_id in out["trail_id"]]
    else:
        fallback_names = [f"trail_{i}" for i in range(len(out))]
    if "name" not in out:
        out["name"] = fallback_names
    else:
        out["name"] = [value if value is not None and not (isinstance(value, float) and math.isnan(value))
                        else fallback_names[i] for i, value in enumerate(out["name"])]
    if area_name is not None:
        out["area"] = area_name
    elif "area" not in out:
        out["area"] = None
    if "source" not in out:
        out["source"] = "TrailNet prediction"
    if "confidence" not in out:
        out["confidence"] = None
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
