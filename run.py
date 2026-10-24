#!/usr/bin/env python3
"""Usage: python run.py [discover|download|frames|mask|sky|sfm|moving|dense|train|all] [--config config.yaml]"""
import argparse, importlib.util, sys
from pathlib import Path

# The stages are a package (relative imports) whose folder may not be named "sdsplat"
# (e.g. a checkout called clerk-splat): register this folder under that name.
_here = Path(__file__).resolve().parent
if "sdsplat" not in sys.modules:
    spec = importlib.util.spec_from_file_location("sdsplat", _here / "__init__.py",
                                                  submodule_search_locations=[str(_here)])
    sys.modules["sdsplat"] = importlib.util.module_from_spec(spec); spec.loader.exec_module(sys.modules["sdsplat"])
from sdsplat import discover, download, frames, mask, sky, sfm, moving, dense, train
from sdsplat.common import load_config

STAGES = {"discover": discover, "download": download, "frames": frames,
          "mask": mask, "sky": sky, "sfm": sfm, "moving": moving, "dense": dense,
          "train": train}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args(); cfg = load_config(a.config)
    for name in (STAGES if a.stage == "all" else [a.stage]):
        print(f"== {name} =="); STAGES[name].main(cfg)
