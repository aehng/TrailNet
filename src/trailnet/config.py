"""Configuration loading and area selection shared by command-line tools."""
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    config_path = Path(path) if path else ROOT / "config.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_root"] = config_path.parent.resolve()
    return config


def resolve_path(config, value):
    path = Path(value)
    return path if path.is_absolute() else config["_root"] / path


def select_areas(config, area=None, all_areas=False):
    areas = config["areas"]
    if all_areas:
        return areas
    if not area:
        raise ValueError("Choose --area NAME or --all-areas")
    matches = [entry for entry in areas if entry["name"] == area]
    if not matches:
        choices = ", ".join(entry["name"] for entry in areas)
        raise ValueError(f"Unknown area {area!r}. Configured areas: {choices}")
    return matches
