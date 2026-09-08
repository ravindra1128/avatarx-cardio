"""Shared config + IO for the cardio-pipeline-optimizer skill.

The skill lives INSIDE the repo it optimizes, so the repo root is derived from this
file's location and nothing here depends on the machine. Override with CARDIO_REPO if
the skill is ever run from elsewhere.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Must match the deployed service's launch overrides, so replay scores what
# production computes. Edit references/decision-rules.md if production changes.
PIPELINE_ENV = {"AFIB_MAX_COLLAPSED_FRACTION": "0.05"}


def repo() -> Path:
    p = os.getenv("CARDIO_REPO")
    if p:
        return Path(p).expanduser().resolve()
    # .../<repo>/.claude/skills/cardio-pipeline-optimizer/scripts/_common.py
    return Path(__file__).resolve().parents[4]


def cache_dir() -> Path:
    d = os.getenv("EVAL_CACHE_DIR") or str(repo() / "data" / "eval_cache")
    p = Path(d).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def corpus_dir() -> Path:
    d = os.getenv("EVAL_CORPUS_DIR") or str(repo() / "data" / "eval_corpus")
    return Path(d).expanduser().resolve()


def manifest_path() -> Path:
    return cache_dir() / "eval_manifest.json"


def apply_pipeline_env() -> None:
    """Set production's launch overrides BEFORE importing the pipeline, and make
    the repo importable. app.measure_overrides reads the env at import time."""
    for k, v in PIPELINE_ENV.items():
        os.environ.setdefault(k, v)
    r = str(repo())
    if r not in sys.path:
        sys.path.insert(0, r)


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _resolve(path: str | Path) -> Path:
    """A bare filename lives in the cache dir; anything with a directory
    component is taken as given (relative to the cwd). Found on the first
    real run: `--out data/eval_cache/tune.json` was nested under the cache
    dir into a directory that did not exist, and the write failed."""
    p = Path(path)
    if not p.is_absolute() and p.parent == Path("."):
        p = cache_dir() / p
    return p


def load_json(path: str | Path) -> dict:
    return json.loads(_resolve(path).read_text())


def dump_json(obj, path: str | Path) -> Path:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p
