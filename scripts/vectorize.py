import argparse
import _bootstrap  # noqa: F401
from trailnet.config import load_config, select_areas
from trailnet.postprocess import vectorize_area


def main():
    parser = argparse.ArgumentParser(description="Vectorize TrailNet probability rasters")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--area")
    group.add_argument("--all-areas", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    for selected in select_areas(config, args.area, args.all_areas):
        print(vectorize_area(config, selected))


if __name__ == "__main__":
    main()
