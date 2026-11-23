"""Load params.yaml once and expose it as PARAMS."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path=None):
    path = Path(path or os.environ.get("BAIROPS_PARAMS", ROOT / "params.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


PARAMS = load_params()
