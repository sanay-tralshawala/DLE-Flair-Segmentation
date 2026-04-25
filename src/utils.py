"""Small shared utilities for config loading and device selection."""


def load_config(config_path) -> dict:
    """Load default.yaml plus a model YAML override into one merged config dict."""
    raise NotImplementedError


def get_device():
    """Return the best available torch device automatically."""
    raise NotImplementedError
