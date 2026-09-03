"""v0.2.0.1 — runtime-dependency preflight: a wrong-interpreter launch
must fail AT STARTUP with the interpreter named and the fix spelled out,
never after the user framed their face (real report: Homebrew Python
with numpy but no cv2 served the whole demo UI and died at scan start
with 'opencv required')."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import cli


def test_preflight_passes_on_this_interpreter():
    assert cli._runtime_preflight(mods=("numpy",)) == 0


def test_preflight_names_interpreter_and_fix(capsys):
    def fake_find(m):
        return None if m == "cv2" else object()
    rc = cli._runtime_preflight(mods=("numpy", "scipy", "cv2"),
                                _find=fake_find)
    err = capsys.readouterr().err
    assert rc == 2
    assert sys.executable in err
    assert "cv2" in err and "opencv-python-headless" in err
    assert "which python3" in err


def test_engine_scan_start_message_names_interpreter(tmp_path, monkeypatch):
    import app.scan_engine as se
    s = se.ScanSession("pf1", str(tmp_path), fps_hint=30.0)
    monkeypatch.setattr(se, "cv2", None)
    s.feedback["ready"] = True
    with pytest.raises(RuntimeError) as e:
        s.start_scan()
    msg = str(e.value)
    assert sys.executable in msg
    assert "opencv-python-headless" in msg


def test_preflight_suggests_a_working_interpreter(capsys, monkeypatch):
    """v0.2.1.1 (third wrong-interpreter launch, via a stray in-tree
    venv): when deps are missing and relaunching is opted out (or already
    happened — AVATARX_NO_REEXEC), the message must end in a copy-paste
    command using a discovered working interpreter."""
    def fake_find(m):
        return None if m == "cv2" else object()
    monkeypatch.setenv("AVATARX_NO_REEXEC", "1")
    monkeypatch.setattr(cli, "_find_working_interpreter",
                        lambda mods: "/usr/bin/python3")
    rc = cli._runtime_preflight(
        mods=("numpy", "cv2"), _find=fake_find,
        _exec=lambda p, a: pytest.fail("must not re-exec when opted out"))
    err = capsys.readouterr().err
    assert rc == 2
    assert "working interpreter WAS found" in err
    assert "/usr/bin/python3" in err
    assert "deactivate" in err                    # venv hint


def test_preflight_relaunches_with_working_interpreter(capsys, monkeypatch):
    """v0.3.0.1 (FOURTH wrong-interpreter launch): the copy-paste
    suggestion demonstrably did not change behaviour — the CLI must
    relaunch itself, once, with the discovered interpreter."""
    def fake_find(m):
        return None if m == "cv2" else object()
    monkeypatch.setenv("AVATARX_NO_REEXEC", "")   # falsy = relaunch armed
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(cli, "_find_working_interpreter",
                        lambda mods: "/usr/bin/python3")
    execs = []
    rc = cli._runtime_preflight(mods=("numpy", "cv2"), _find=fake_find,
                                _exec=lambda p, a: execs.append((p, a)))
    assert execs == [("/usr/bin/python3",
                      ["/usr/bin/python3"] + sys.argv)]
    assert rc == 0                       # unreachable under a real exec
    err = capsys.readouterr().err
    assert "relaunching with /usr/bin/python3" in err
    # the loop guard is now set: a second attempt must NOT exec again
    assert cli.os.environ["AVATARX_NO_REEXEC"] == "1"
    rc2 = cli._runtime_preflight(
        mods=("numpy", "cv2"), _find=fake_find,
        _exec=lambda p, a: pytest.fail("re-exec loop"))
    assert rc2 == 2


def test_find_working_interpreter_prefers_first_passing():
    got = cli._find_working_interpreter(
        ("numpy",),
        _candidates=["/nonexistent/py", sys.executable, "/usr/bin/python3"],
        _check=lambda c: c == "/usr/bin/python3")
    assert got == "/usr/bin/python3"
    assert cli._find_working_interpreter(
        ("numpy",), _candidates=["/usr/bin/python3"],
        _check=lambda c: False) is None
