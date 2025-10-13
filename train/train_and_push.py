#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from git import Repo

# KitOps imports
from kitops.modelkit.kitfile import Kitfile

# -----------------------
# Config
# -----------------------
KIT_MODEL_NAME = os.getenv("KIT_MODEL_NAME", "testmodelkit")
GIT_REPO_URL = os.getenv("GIT_REPO_URL", "https://github.com/Chetand411/gitops-public.git")
GIT_TOKEN = os.getenv("GIT_TOKEN")
JOZU_USERNAME = os.getenv("JOZU_USERNAME")
JOZU_PASSWORD = os.getenv("JOZU_PASSWORD")
MANIFEST_FILE = Path("/workspace/manifests/model-version.yaml")
WORKSPACE = Path("/workspace")
MODELKIT_DIR = WORKSPACE / "modelkit"

# -----------------------
# Helpers
# -----------------------
def run_cmd(cmd, check=True):
    print(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"[ERROR] {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout.strip()

# -----------------------
# 1. Prepare workspace
# -----------------------
MODELKIT_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------
# 2. Train Iris Model
# -----------------------
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
model_file = MODELKIT_DIR / "model.pkl"
joblib.dump(model, model_file)
print(f"[INFO] Model trained with accuracy: {acc:.2f}")

# -----------------------
# 3. Generate Kitfile
# -----------------------
version = "1.0.1"
kitfile = Kitfile()
kitfile.manifestVersion = "1.0"
kitfile.package = {
    "name": KIT_MODEL_NAME,
    "version": version,
    "description": "Iris classifier packaged as ModelKit",
    "authors": ["Naga"]
}
kitfile.model = {
    "name": KIT_MODEL_NAME,
    "path": str(model_file),
    "license": "Apache 2.0",
    "version": version,
    "description": "Trained Iris Logistic Regression model",
    "parts": []
}
kitfile.datasets = [
    {
        "name": "iris",
        "path": "./data",
        "license": "Public Domain",
        "version": "1.0",
        "description": "Iris dataset used for training"
    }
]
kitfile.code = [
    {
        "name": "train script",
        "path": "./train_and_push.py",
        "description": "Python script used to train the model"
    }
]
kitfile.docs = [
    {
        "name": "README",
        "path": "./docs/README.md",
        "description": "Documentation for the ModelKit"
    }
]

kitfile_path = MODELKIT_DIR / f"{KIT_MODEL_NAME}.kit"
with open(kitfile_path, "w") as f:
    f.write(kitfile.to_yaml())
print(f"[INFO] Kitfile created at {kitfile_path}")

# -----------------------
# 4. Download and install KitOps CLI
# -----------------------
run_cmd("curl -L https://github.com/kitops-ml/kitops/releases/download/v1.8.0/kitops-linux-x86_64.tar.gz -o kitops.tar.gz")
run_cmd("tar -xzf kitops.tar.gz")
run_cmd("chmod +x kit")
run_cmd("mv kit /usr/local/bin/kit")
print("[INFO] KitOps CLI installed")

# -----------------------
# 5. Login to Jozu
# -----------------------
run_cmd(f"kit login jozu.ml --username {JOZU_USERNAME} --password {JOZU_PASSWORD}")

# -----------------------
# 6. Initialize Kit (optional, safe if already exists)
# -----------------------
run_cmd(f"kit init {MODELKIT_DIR} --name {KIT_MODEL_NAME} --desc 'Iris ModelKit'")

# -----------------------
# 7. Package Model
# -----------------------
run_cmd(f"cd {MODELKIT_DIR} && kit pack . -t jozu.ml/naga.linux17/{KIT_MODEL_NAME}:{version}")

# -----------------------
# 8. Push Model to Registry
# -----------------------
run_cmd(f"kit push jozu.ml/naga.linux17/{KIT_MODEL_NAME}:{version}")
print(f"[INFO] Model pushed to jozu.ml with version {version}")

# -----------------------
# 9. Update Manifest and Push to Git
# -----------------------
if MANIFEST_FILE.exists():
    import yaml
    with open(MANIFEST_FILE, "r") as f:
        manifest = yaml.safe_load(f)
    manifest['version'] = version
    with open(MANIFEST_FILE, "w") as f:
        yaml.safe_dump(manifest, f)

    repo = Repo(WORKSPACE)
    origin = repo.remote()
    origin.set_url(f"https://{GIT_TOKEN}@github.com/Chetand411/gitops-public.git")
    repo.git.add([str(MANIFEST_FILE)])
    repo.index.commit(f"Update {KIT_MODEL_NAME} version to {version}")
    repo.remote().push()
    print("[INFO] Manifest updated and pushed to Git")

