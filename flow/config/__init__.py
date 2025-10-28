import logging
import importlib.resources as resources
import yaml


logger = logging.getLogger(__name__)


def load_config(filename: str) -> dict:
    """Load YAML config from config directory."""
    if not filename or any(sep in filename for sep in ("/", "\\")):
        raise ValueError(f"Provide a filename (e.g. 'train.yaml') not a path; got {filename}")
    if not filename.lower().endswith((".yaml", ".yml")):
        raise ValueError(f"Config filename must end with .yaml or .yml; got {filename}")

    pkg = __package__
    file = resources.files(pkg).joinpath(filename)
    logger.info(f"Loading config from {file}")
    with open(file, encoding="utf-8") as f:
        return yaml.safe_load(f)
