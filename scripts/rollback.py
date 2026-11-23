"""Roll back production to an earlier registered version (default: the previous one)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bairops import registry  # noqa: E402
from bairops.config import PARAMS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--version", default="", help="version to restore; empty = previous version")
a = ap.parse_args()
name = PARAMS["project"]["model_name"]
cur = registry.production_version(name)
if a.version:
    target = a.version
else:
    older = [v for v in registry.versions(name) if int(v.version) < int(cur.version)]
    if not older:
        sys.exit(f"No version older than production v{cur.version} to roll back to.")
    target = older[-1].version
registry.promote(name, target)
print(f"Rolled back: production v{cur.version} -> v{target}")
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(f"## BAirOps rollback\n\nProduction moved from **v{cur.version}** to **v{target}**.\n")
