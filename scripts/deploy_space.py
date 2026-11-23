"""Deploy (or restart) the BAirOps API + dashboard on a free Hugging Face Docker Space.

Env: HF_TOKEN, HF_SPACE (e.g. "yourname/bairops"), MLFLOW_TRACKING_URI/USERNAME/PASSWORD.
  python scripts/deploy_space.py            # upload code + restart
  python scripts/deploy_space.py --restart  # only restart (loads the current production model)
"""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--restart", action="store_true")
a = ap.parse_args()

space = os.environ["HF_SPACE"]
api = HfApi(token=os.environ["HF_TOKEN"])

if a.restart:
    api.restart_space(space)
    print(f"Restarted https://huggingface.co/spaces/{space}")
    raise SystemExit(0)

api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
    api.add_space_secret(space, key, os.environ[key])
api.add_space_variable(space, "BAIROPS_GIT_SHA", os.environ.get("GITHUB_SHA", "local")[:7])

root = os.path.join(os.path.dirname(__file__), "..")
with tempfile.TemporaryDirectory() as tmp:
    for item in ("Dockerfile", "requirements-api.txt", "params.yaml", "src", "app"):
        s, d = os.path.join(root, item), os.path.join(tmp, item)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, d)
    shutil.copy(os.path.join(root, "space", "README.md"), os.path.join(tmp, "README.md"))
    for junk in ("__pycache__",):
        for dp, dn, _ in os.walk(tmp):
            if junk in dn:
                shutil.rmtree(os.path.join(dp, junk))
    api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                      commit_message=f"deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}")
print(f"Deployed -> https://huggingface.co/spaces/{space}  (app: https://{space.replace('/', '-').lower()}.hf.space)")
