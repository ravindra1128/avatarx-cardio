"""M2.7 — campaign manifests: quota fill, consent-version and checklist
integrity, CLI status."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess

import pytest

from datasets.campaigns import campaign_status, load_campaign, CampaignError

_ROOT = pathlib.Path(__file__).resolve().parents[1]

SPEC = """campaign:
  id: camp-test-1
  version: 1
  consent_form_version: v2.1
  target_n: 4
  quotas:
    fitzpatrick: {I: 1, IV: 1, VI: 1}
    devices: {iphone_15: 2, pixel_9: 1}
    lighting: {indoor_bright: 2, indoor_dim: 1}
  session_checklist: [consent_signed, prbs_sync_run, ecg_leads_verified]
"""


def _session(pid, fitz, device, lighting, consent="v2.1", checklist=None):
    return {"participant_id": pid, "fitzpatrick": fitz, "device": device,
            "lighting": lighting, "consent_form_version": consent,
            "checklist": checklist or {"consent_signed": True,
                                       "prbs_sync_run": True,
                                       "ecg_leads_verified": True}}


def _write(tmp, sessions):
    (tmp / "campaign.yaml").write_text(SPEC)
    sdir = tmp / "sessions"
    sdir.mkdir(exist_ok=True)
    for i, s in enumerate(sessions):
        (sdir / f"s{i:03d}.session.json").write_text(json.dumps(s))


def test_quota_fill_counts_participants_and_sessions(tmp_path):
    _write(tmp_path, [
        _session("p1", "I", "iphone_15", "indoor_bright"),
        _session("p1", "I", "iphone_15", "indoor_dim"),   # same participant
        _session("p2", "IV", "pixel_9", "indoor_bright"),
    ])
    st = campaign_status(tmp_path)
    assert st.enrolled_participants == 2 and st.sessions == 3
    assert st.quota_fill["fitzpatrick"]["I"] == {"target": 1, "have": 1}
    assert st.quota_fill["fitzpatrick"]["VI"]["have"] == 0
    assert st.quota_fill["devices"]["iphone_15"]["have"] == 2   # per session
    assert any(u.startswith("fitzpatrick:VI") for u in st.unmet)
    assert not st.problems and not st.complete()


def test_integrity_findings_consent_and_checklist(tmp_path):
    _write(tmp_path, [
        _session("p1", "I", "iphone_15", "indoor_bright", consent="v1.0"),
        _session("p2", "IV", "pixel_9", "indoor_bright",
                 checklist={"consent_signed": True, "prbs_sync_run": False,
                            "ecg_leads_verified": True}),
    ])
    st = campaign_status(tmp_path)
    assert st.enrolled_participants == 0            # neither session valid
    assert len(st.problems) == 2
    assert any("consent form" in p for p in st.problems)
    assert any("checklist incomplete" in p for p in st.problems)


def test_bad_manifest_fails_closed(tmp_path):
    (tmp_path / "campaign.yaml").write_text("campaign: {id: x}\n")
    with pytest.raises(CampaignError):
        campaign_status(tmp_path)


def test_cli_campaign_status(tmp_path):
    _write(tmp_path, [_session("p1", "I", "iphone_15", "indoor_bright")])
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"), "campaign",
                        "status", str(tmp_path)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["campaign_id"] == "camp-test-1"
    assert doc["enrolled_participants"] == 1 and doc["complete"] is False


# ---------------------------------------------- v0.4-vascular additions
def test_vascular_campaign_axes_and_retest(tmp_path):
    import shutil
    from datasets.campaigns import campaign_status, load_campaign
    src = pathlib.Path(__file__).resolve().parents[1] / "configs" \
        / "campaign_vascular_template.yaml"
    camp = load_campaign(src)                      # template is valid
    assert "age_band" in camp["quotas"] and "bp_range" in camp["quotas"]
    d = tmp_path / "camp"
    (d / "sessions").mkdir(parents=True)
    shutil.copy(src, d / "campaign.yaml")
    base = {"consent_form_version": "consent-vasc-v1",
            "fitzpatrick": "III", "device": "rig-1",
            "lighting": "controlled", "age_band": "40-59",
            "bp_range": "normotensive",
            "checklist": {c: True for c in camp["session_checklist"]}}
    for i, name in enumerate(["a1", "a2", "b1"]):
        rec = dict(base, participant_id="pa" if name.startswith("a")
                   else "pb")
        (d / "sessions" / f"{name}.session.json").write_text(
            json.dumps(rec))
    st = campaign_status(d)
    assert st.retest_pairs == 1                    # pa has two sessions
    assert any(u.startswith("retest pairs") for u in st.unmet)
    assert st.quota_fill["age_band"]["40-59"]["have"] == 2
    assert st.quota_fill["bp_range"]["normotensive"]["have"] == 2


def test_unknown_quota_axis_fails_closed(tmp_path):
    from datasets.campaigns import CampaignError, load_campaign
    src = pathlib.Path(__file__).resolve().parents[1] / "configs" \
        / "campaign_vascular_template.yaml"
    text = src.read_text().replace("age_band:", "vibe_check:")
    p = tmp_path / "c.yaml"
    p.write_text(text)
    import pytest
    with pytest.raises(CampaignError, match="vibe_check"):
        load_campaign(p)


def test_vasotone_campaign_template_and_maneuver_axis(tmp_path):
    import shutil
    from datasets.campaigns import campaign_status, load_campaign
    src = pathlib.Path(__file__).resolve().parents[1] / "configs" \
        / "campaign_vasotone_template.yaml"
    camp = load_campaign(src)
    assert "maneuver" in camp["quotas"]
    assert "null_optics" in camp["quotas"]["maneuver"]
    d = tmp_path / "camp"
    (d / "sessions").mkdir(parents=True)
    shutil.copy(src, d / "campaign.yaml")
    base = {"participant_id": "pa",
            "consent_form_version": "consent-tone-v1",
            "fitzpatrick": "III", "device": "rig-1",
            "lighting": "controlled", "maneuver": "cold_pressor",
            "checklist": {c: True for c in camp["session_checklist"]}}
    (d / "sessions" / "a1.session.json").write_text(json.dumps(base))
    (d / "sessions" / "a2.session.json").write_text(json.dumps(
        dict(base, maneuver="null_optics")))
    st = campaign_status(d)
    assert st.quota_fill["maneuver"]["cold_pressor"]["have"] == 1
    assert st.quota_fill["maneuver"]["null_optics"]["have"] == 1
    assert any(u.startswith("maneuver:null_rest") for u in st.unmet)


# ------------------------------------------------- v0.6 flutter axes
def test_flutter_campaign_axes_and_series_subprotocol(tmp_path):
    """rhythm_class / conduction_ratio quotas and the >= 3-scan series
    sub-protocol the serial gate (F3) needs — a retest PAIR is not a
    series."""
    from datasets.campaigns import campaign_status
    d = tmp_path / "camp"
    (d / "sessions").mkdir(parents=True)
    (d / "campaign.yaml").write_text("""
campaign:
  id: flut
  version: 1
  consent_form_version: c1
  target_n: 2
  quotas:
    fitzpatrick: {VI: 1}
    devices: {rig-1: 1}
    lighting: {controlled: 1}
    rhythm_class: {atrial_flutter: 1, sinus_tachycardia: 1}
    conduction_ratio: {TWO_TO_ONE: 1, FOUR_TO_ONE: 1}
  series: {target_series: 1, min_scans: 3}
  session_checklist: [twelve_lead_ecg_recorded]
""")

    def _sess(name, pid, *, rhythm, ratio, checklist=True, fitz="VI"):
        (d / "sessions" / f"{name}.session.json").write_text(json.dumps({
            "participant_id": pid, "fitzpatrick": fitz, "device": "rig-1",
            "lighting": "controlled", "consent_form_version": "c1",
            "rhythm_class": rhythm, "conduction_ratio": ratio,
            "checklist": {"twelve_lead_ecg_recorded": checklist}}))

    _sess("s1", "p1", rhythm="atrial_flutter", ratio="TWO_TO_ONE")
    _sess("s2", "p1", rhythm="atrial_flutter", ratio="FOUR_TO_ONE")
    st = campaign_status(d)
    assert st.problems == []
    assert st.quota_fill["rhythm_class"]["atrial_flutter"]["have"] == 2
    assert st.quota_fill["conduction_ratio"]["TWO_TO_ONE"]["have"] == 1
    # two scans is a retest PAIR but not yet a series
    assert st.retest_pairs == 1
    assert st.scan_series == 0
    assert any("scan series" in u for u in st.unmet)
    _sess("s3", "p1", rhythm="sinus_tachycardia", ratio="TWO_TO_ONE")
    st2 = campaign_status(d)
    assert st2.scan_series == 1
    assert not any("scan series" in u for u in st2.unmet)


def test_a_session_missing_a_quotad_flutter_field_fails_closed(tmp_path):
    """The v0.5 review's finding, re-checked for the new axes: a session
    lacking a quota'd field is a PROBLEM, never a KeyError."""
    from datasets.campaigns import campaign_status
    d = tmp_path / "camp2"
    (d / "sessions").mkdir(parents=True)
    (d / "campaign.yaml").write_text("""
campaign:
  id: flut2
  version: 1
  consent_form_version: c1
  target_n: 1
  quotas:
    fitzpatrick: {VI: 1}
    devices: {rig-1: 1}
    lighting: {controlled: 1}
    rhythm_class: {atrial_flutter: 1}
  session_checklist: [twelve_lead_ecg_recorded]
""")
    (d / "sessions" / "s1.session.json").write_text(json.dumps({
        "participant_id": "p1", "fitzpatrick": "VI", "device": "rig-1",
        "lighting": "controlled", "consent_form_version": "c1",
        "checklist": {"twelve_lead_ecg_recorded": True}}))
    st = campaign_status(d)
    assert any("rhythm_class" in p for p in st.problems)
    assert st.enrolled_participants == 0


def test_the_shipped_flutter_template_loads_and_quotas_the_known_miss():
    from datasets.campaigns import load_campaign
    camp = load_campaign(_ROOT / "configs" /
                         "campaign_flutter_template.yaml")
    q = camp["quotas"]
    # both arms present, and the negatives dominate
    assert q["rhythm_class"]["sinus_tachycardia"] > \
        q["rhythm_class"]["atrial_flutter"]
    # every ratio the F2 gate requires is quota'd, incl. the known miss
    assert set(q["conduction_ratio"]) == {"TWO_TO_ONE", "THREE_TO_ONE",
                                          "FOUR_TO_ONE", "VARIABLE"}
    assert q["conduction_ratio"]["FOUR_TO_ONE"] >= 10
    # the darkest band is quota'd rather than hoped for
    assert q["fitzpatrick"]["VI"] > 0
    assert camp["series"]["min_scans"] >= 3
    # the label bar is on the session checklist
    for item in ("twelve_lead_ecg_recorded", "ep_adjudication_recorded",
                 "conduction_ratio_recorded"):
        assert item in camp["session_checklist"], item


def test_axis_key_map_covers_every_declared_axis():
    """The map and the axis tuples must not drift: a quota axis with no
    session field would make the status reporter ignore it."""
    from datasets.campaigns import (AXIS_KEY, OPTIONAL_QUOTA_AXES,
                                    QUOTA_AXES, _MISSING_AXIS_KEYS)
    assert set(AXIS_KEY) == set(QUOTA_AXES) | set(OPTIONAL_QUOTA_AXES)
    assert _MISSING_AXIS_KEYS == set()
