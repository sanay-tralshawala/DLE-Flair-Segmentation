"""Small shared utilities for config loading and device selection."""

from copy import deepcopy
from pathlib import Path
from math import isnan

import torch
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override values into base yaml config without mutating either input."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict:
    with open(path) as file:
        return yaml.safe_load(file) or {}


def load_config(config_path) -> dict:
    """Load a YAML config and recursively merge any files listed under `defaults`."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve()

    config = _load_yaml(config_path)
    defaults = config.pop("defaults", [])
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]

    merged = {}
    for default_path in defaults:
        default_path = Path(default_path)
        if not default_path.is_absolute():
            default_path = config_path.parent / default_path
        merged = _deep_merge(merged, load_config(default_path))

    return _deep_merge(merged, config)


def get_device():
    """Return the best available torch device automatically."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # compatibility for Apple Silicon (henry's laptop)
        return torch.device("mps")
    return torch.device("cpu")


def format_table_value(value, digits: int = 4) -> str:
    """Format notebook table values consistently."""
    if isinstance(value, float):
        return "nan" if isnan(value) else f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], digits: int = 4) -> None:
    """Display a compact Markdown table in a notebook."""
    from IPython.display import Markdown, display

    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values = [
            format_table_value(row.get(key, ""), digits=digits)
            for key, _ in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    display(Markdown("\n".join(lines)))


def zero_channel_transform(channel_indices: list[int]):
    """Return a transform that zeroes selected image channels in a batch."""
    channel_indices = list(channel_indices)
    if not channel_indices:
        return None

    def transform(images: torch.Tensor) -> torch.Tensor:
        images = images.clone()
        images[:, channel_indices] = 0.0
        return images

    return transform
