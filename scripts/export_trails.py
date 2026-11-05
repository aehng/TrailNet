import argparse
import _bootstrap  # noqa: F401
from trailnet.config import load_config, select_areas
from trailnet.export import FORMATS, export_area

parser = argparse.ArgumentParser(description="Export predicted trails to GPX, KML, or GeoJSON")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--area")
group.add_argument("--all-areas", action="store_true")
parser.add_argument("--config")
parser.add_argument("--format", choices=FORMATS, default="gpx")
args = parser.parse_args()
config = load_config(args.config)
for selected in select_areas(config, args.area, args.all_areas):
    print(export_area(config, selected, args.format))
