import json, re, shutil, subprocess
from pathlib import Path
import yaml

def load_config(path="config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["workspace"] = Path(cfg["workspace"]).resolve() / cfg["street"]
    cfg["workspace"].mkdir(parents=True, exist_ok=True)
    return cfg

def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80]

def run(cmd, **kw):
    print("  $", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, **kw)

def add_cuda_dlls():
    """Windows COLMAP's ONNX features (ALIKED/LightGlue) need CUDA 12 DLLs; the nvidia-*-cu12
    pip packages ship them under site-packages/nvidia/*/bin. Put those on PATH for children."""
    import os, importlib.util
    if os.name != "nt" or not importlib.util.find_spec("nvidia"): return
    root = Path(importlib.util.find_spec("nvidia").submodule_search_locations[0])
    bins = [str(p) for p in root.glob("*/bin") if p.is_dir()]
    os.environ["PATH"] = os.pathsep.join(bins + [os.environ.get("PATH", "")])

def need(binary):
    if not shutil.which(binary):
        raise SystemExit(f"'{binary}' not found on PATH — see README install steps.")

def manifest_path(cfg):
    return cfg["workspace"] / "manifest.json"

def load_manifest(cfg):
    p = manifest_path(cfg)
    return json.loads(p.read_text()) if p.exists() else []

def save_manifest(cfg, items):
    manifest_path(cfg).write_text(json.dumps(items, indent=2))

def segment_dirs(cfg):
    return sorted(p for p in (cfg["workspace"] / "segments").glob("*") if p.is_dir())
