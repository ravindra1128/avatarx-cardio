"""Shared fixtures: the synthetic video set is built once per session."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture(scope="session")
def videos(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    from scripts.make_synth_video import synth_video
    d = tmp_path_factory.mktemp("synthv")
    out = {}
    specs = [
        ("sinus30", dict(kind="sinus", fps=30.0)),
        ("sinus60", dict(kind="sinus", fps=60.0)),
        ("af30",    dict(kind="af", fps=30.0)),
        ("af60",    dict(kind="af", fps=60.0)),
        ("dark30",  dict(kind="sinus", fps=30.0, lux_scale=0.15)),
        ("noface30", dict(kind="sinus", fps=30.0, face=False)),
    ]
    for name, kw in specs:
        path = str(d / f"{name}.avi")
        truth = synth_video(path, duration_s=20.0, seed=7, **kw)
        out[name] = (path, truth)
    return out
