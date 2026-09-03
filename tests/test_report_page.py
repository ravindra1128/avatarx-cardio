"""v0.2.1 Task 1 — the ad-hoc result graphs are GONE from the demo page
(metric grid, Rhythm Map panels, legacy waveform canvas, rate-flag
injection); the unified report container replaced them."""
import pathlib

_HTML = (pathlib.Path(__file__).resolve().parents[1] / "app" / "static"
         / "index.html").read_text()


def test_old_result_render_paths_are_gone():
    for token in ('id="wave"', 'id="rhythmMap"', 'id="rmTitle"',
                  'id="rGrid"', 'id="rText"', "drawWave",
                  "rhythm_map_svg", "rate_flag_sentence"):
        assert token not in _HTML, token


def test_report_container_and_print_present():
    assert 'id="report"' in _HTML
    assert "window.print()" in _HTML
    assert 'id="rDetails"' in _HTML          # explainability panel kept
