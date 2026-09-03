"""
Configuration loading (single source of truth: configs/default.yaml).

A deliberately SMALL YAML-subset parser lives here instead of a PyYAML
dependency: the v0.1 dependency list is pinned (numpy, scipy,
opencv-python-headless, mediapipe, pytest) and the config format is under
our control, so the subset is a contract, not a limitation. Supported:

  * indentation-nested mappings
  * inline mappings  {a: 1, b: two}
  * inline lists     [1, 2.5, three]
  * scalars: int, float, true/false booleans, everything else a string
    ("CALIBRATED" stays the sentinel string the spec uses; "off" stays a
    string rather than YAML 1.1's surprising boolean coercion)
  * comments (#) and blank lines

Anything outside the subset raises — a config that parses differently than
it reads would corrupt every provenance hash downstream.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Optional

_DEFAULT_PATH = pathlib.Path(__file__).parent / "default.yaml"


def _scalar(tok: str) -> Any:
    t = tok.strip()
    if t == "":
        return None
    if t in ("true", "True"):
        return True
    if t in ("false", "False"):
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t.strip("'\"")


def _split_top(s: str, sep: str) -> list[str]:
    """Split on `sep` outside brackets."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _value(tok: str) -> Any:
    t = tok.strip()
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        return [] if not inner else [_value(p) for p in _split_top(inner, ",")]
    if t.startswith("{") and t.endswith("}"):
        d: dict = {}
        inner = t[1:-1].strip()
        if inner:
            for part in _split_top(inner, ","):
                if ":" not in part:
                    raise ValueError(f"bad inline mapping entry: {part!r}")
                k, v = part.split(":", 1)
                d[k.strip()] = _value(v)
        return d
    return _scalar(t)


def _logical_lines(text: str) -> list[str]:
    """Strip comments/blanks and merge lines while brackets stay open, so
    the spec's wrapped inline collections parse as written."""
    out: list[str] = []
    buf = ""
    depth = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        buf = (buf + " " + line.strip()) if buf else line
        depth = buf.count("[") + buf.count("{") \
            - buf.count("]") - buf.count("}")
        if depth > 0:
            continue
        out.append(buf)
        buf = ""
    if buf:
        raise ValueError(f"unbalanced brackets at end of config: {buf!r}")
    return out


def parse_yaml_subset(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in _logical_lines(text):
        line = raw
        indent = len(line) - len(line.lstrip())
        if "\t" in raw[:indent]:
            raise ValueError("tabs are not allowed in indentation")
        key_part = line.strip()
        if ":" not in key_part:
            raise ValueError(f"expected 'key:' in line: {raw!r}")
        key, rest = key_part.split(":", 1)
        key, rest = key.strip(), rest.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _value(rest)
    return root


def load_config(path: Optional[str] = None) -> dict:
    p = pathlib.Path(path) if path else _DEFAULT_PATH
    return parse_yaml_subset(p.read_text())


def config_hash(cfg: dict) -> str:
    """Provenance hash of the RESOLVED config (carried into ScanResult)."""
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
