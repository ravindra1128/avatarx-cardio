"""
§F evaluation harness (v0.6 flutter track) — the hard-negative battery.

The product risk of this track is not missing flutter. It is FLAGGING
ORDINARY PEOPLE: a sustained regular tachycardia is usually sinus, and
at screening prevalence a flag with mediocre specificity is a
false-referral engine. So this module is built around the negatives —
every confounder is a labeled class with its own reported false-positive
rate, and the specificity gate (F1) is computed against the battery, not
against clean sinus rhythm.

Four baselines the flag must be measured against, in increasing honesty:

  B1  rate only ("> 140 bpm => flag") — the strawman that shows how much
      of any apparent performance is just "fast".
  B2  rate + naive fixed dispersion.
  B3  rate band + age-adjusted dispersion floor + respiratory coupling —
      THE INTERPRETABLE COMPETITOR. It is the head's own rule without
      the head's abstention discipline. If the head cannot beat it, B3
      ships, and the scoreboard says so in those words.
  B4  demographics + heart rate (age, sex, rate) — the "is it just who
      the patient is and how fast their heart goes" control.

Comparisons are at MATCHED SENSITIVITY on participant-disjoint splits.
The head's structural advantage over B3 is that it ABSTAINS where B3
guesses, so its specificity is always reported beside its no-read rate:
a rule that declines half the cohort has not earned a better number.

Nothing here reads the reconstruction track (F-d). Rows carry features
already computed by features/flutter.py on the production path.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from evaluation.afib_metrics import (evaluate_binary, projected_ppv,
                                     wilson_ci)
from evaluation.flutter_gates import (append_scoreboard, DEFAULT_RUNS,
                                      load_flutter_gates, shipping_rule)
from features.flutter import (age_band, DEFAULT_RMSSD_FLOOR_MS,
                              latent_atrial_fit, measurement_floor_ms)

# Deployment prevalences for the Task-6 projection. PLANNING VALUES —
# AF is the screening-population anchor; sustained flutter is roughly an
# order of magnitude rarer, and regular SVT rarer still.
DEFAULT_PREVALENCE = {"afib": 0.03, "flutter": 0.003}
# A referral candidate that catches almost nothing has the best false-
# alarm burden trivially, so the burden ranking applies only above this
# sensitivity. PLANNING VALUE.
MIN_ENDPOINT_SENSITIVITY = 0.50
SEED = 20260901


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _feat(row, *path, default=None):
    cur = row.get("features") or {}
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def is_flutter(row) -> bool:
    return str(row.get("rhythm") or "") == "ATRIAL_FLUTTER"


def is_target(row) -> bool:
    """The flag's declared TARGET: 2:1-conducted flutter. Other ratios
    are still flutter and still reported (F2) — they are simply not what
    this flag claims to catch."""
    return is_flutter(row) and \
        str(row.get("conduction_ratio") or "") == "TWO_TO_ONE"


# ------------------------------------------------------------- rules
def b1_rate_only(row, *, bpm_min: float = 140.0):
    bpm = _f(_feat(row, "rate", "median_bpm"))
    return None if bpm is None else bool(bpm > bpm_min)


def b2_rate_and_naive_dispersion(row, *, bpm_min: float = 140.0,
                                 cv_max: float = 0.05):
    bpm = _f(_feat(row, "rate", "median_bpm"))
    cv = _f(_feat(row, "regularity", "cv"))
    if bpm is None or cv is None:
        return None
    return bool(bpm > bpm_min and cv < cv_max)


def b3_interpretable(row, *, coupling_max: float = 0.20,
                     floors=None, bands=("2:1",)):
    """The competitor to beat: the same three conditions the head uses,
    but it never abstains — a missing coupling measurement is read as
    "not coupled", which is exactly the optimistic assumption the head
    refuses to make."""
    rate = (row.get("features") or {}).get("rate") or {}
    reg = (row.get("features") or {}).get("regularity") or {}
    bpm = _f(rate.get("median_bpm"))
    if bpm is None:
        return None
    in_band = bool(rate.get("in_band") and rate.get("band") in bands)
    floors = dict(floors or DEFAULT_RMSSD_FLOOR_MS)
    band = age_band(row.get("age_years"))
    floor = float(floors.get(band, floors["unknown"]))
    m_floor = measurement_floor_ms(_feat(row, "fps") or row.get("fps"))
    if m_floor is not None:
        floor = max(floor, m_floor)
    rmssd = _f(reg.get("rmssd_ms"))
    below = bool(rmssd is not None and rmssd < floor)
    coup = _f((reg.get("coupling") or {}).get("tachogram_resp_fraction"))
    decoupled = True if coup is None else bool(coup <= coupling_max)
    return bool(in_band and below and decoupled)


def head_flag(row):
    """The head's own verdict as recorded on the row: True/False, or
    None where it abstained."""
    v = row.get("head_flag", "__missing__")
    if v == "__missing__":
        return None
    return None if v is None else bool(v)


def _b3_score(row, *, floors=None, bands=("2:1",)):
    """A sweepable score for B3: 1 - coupling fraction, zeroed unless the
    two hard conditions hold. Lets the rule be compared at a matched
    sensitivity instead of only at its default operating point."""
    if b3_interpretable(row, coupling_max=1.01, floors=floors,
                        bands=bands) is not True:
        return 0.0
    coup = _f(_feat(row, "regularity", "coupling",
                    "tachogram_resp_fraction"))
    return 1.0 if coup is None else float(max(0.0, 1.0 - coup))


def _rates(rows, verdicts):
    """(sensitivity, specificity, n_judged) for boolean verdicts, with
    None treated as "not judged" and excluded from both."""
    tp = fp = tn = fn = 0
    for r, v in zip(rows, verdicts):
        if v is None:
            continue
        if is_target(r):
            tp += int(bool(v))
            fn += int(not v)
        else:
            fp += int(bool(v))
            tn += int(not v)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return sens, spec, tp + fp + tn + fn


def b4_demographics_and_rate(rows_train, rows_test):
    """Logistic on (age, sex, rate). Fit on TRAIN participants only."""
    from models.baseline import LogisticModel

    def _X(rows):
        out = []
        for r in rows:
            age = _f(r.get("age_years"))
            sex = 1.0 if str(r.get("sex") or "").upper().startswith("F") \
                else 0.0
            bpm = _f(_feat(r, "rate", "median_bpm")) or 0.0
            out.append([age if age is not None else 55.0, sex, bpm])
        return np.asarray(out, float)

    y = np.asarray([1 if is_target(r) else 0 for r in rows_train], int)
    if len(np.unique(y)) < 2:
        return None
    m = LogisticModel().fit(_X(rows_train), y)
    return np.asarray(m.predict_proba(_X(rows_test)), float)


# ---------------------------------------------------- F0 rate accuracy
def rate_accuracy(rows) -> dict:
    """Pulse-rate error vs the ECG reference DURING arrhythmia — the
    precondition for any rate-based flag. Sinus scans are excluded on
    purpose: the reference study's -7.45 bpm bias was an arrhythmia
    number, and averaging in easy sinus scans would hide it."""
    errs, n_within = [], 0
    for r in rows:
        # SINUS_TACHYCARDIA is a sinus rhythm — counting the track's
        # dominant confounder as an "arrhythmia scan" dilutes exactly
        # the bias this gate exists to bound (review finding). An
        # uninterpretable label is not a reference either.
        if str(r.get("rhythm") or "") in ("SINUS", "SINUS_BRADYCARDIA",
                                          "SINUS_TACHYCARDIA",
                                          "RESPIRATORY_SINUS_ARRHYTHMIA",
                                          "UNINTERPRETABLE", "", "None"):
            continue
        ref = _f(r.get("ecg_rate_bpm"))
        got = _f(_feat(r, "rate", "median_bpm"))
        if ref is None or got is None:
            continue
        errs.append(got - ref)
        n_within += int(abs(got - ref) <= 5.0)
    if not errs:
        return {"n_arrhythmia_scans": 0, "bias_bpm": None,
                "within_5bpm_fraction": None,
                "reason": "no arrhythmia scan carried a paired ECG rate"}
    e = np.asarray(errs, float)
    return {"n_arrhythmia_scans": int(e.size),
            "bias_bpm": round(float(np.mean(e)), 3),
            "mae_bpm": round(float(np.mean(np.abs(e))), 3),
            "p95_abs_error_bpm": round(float(np.percentile(np.abs(e),
                                                           95)), 3),
            "within_5bpm_fraction": round(n_within / e.size, 4),
            "within_5bpm_ci95": [round(x, 4) for x in
                                 wilson_ci(n_within, int(e.size))]}


# ------------------------------------------------- the negative battery
def hard_negative_battery(rows, rule=head_flag) -> dict:
    """Per-confounder false-positive rates. Every negative class is
    named and counted separately: a pooled specificity hides exactly the
    class that will generate the referrals."""
    per = {}
    for r in rows:
        if is_flutter(r):
            continue
        # the class key is the ADJUDICATED RHYTHM unless the cohort
        # carries a finer label (e.g. exertional vs febrile sinus
        # tachycardia). Lower-cased so the gates' pre-registered
        # `require_dominant_negative` name matches what is reported —
        # a spelling mismatch there would red the gate forever with
        # "no share recorded" and look like missing data.
        cls = str(r.get("cohort") or r.get("rhythm") or "unknown").lower()
        d = per.setdefault(cls, {"n": 0, "n_flagged": 0,
                                 "n_abstained": 0})
        d["n"] += 1
        v = rule(r)
        if v is None:
            d["n_abstained"] += 1
        elif v:
            d["n_flagged"] += 1
    for cls, d in per.items():
        judged = d["n"] - d["n_abstained"]
        d["false_positive_rate"] = (round(d["n_flagged"] / judged, 4)
                                    if judged else None)
        d["abstention_rate"] = (round(d["n_abstained"] / d["n"], 4)
                                if d["n"] else None)
        d["fp_ci95"] = ([round(x, 4) for x in
                         wilson_ci(d["n_flagged"], judged)]
                        if judged else None)
    total = sum(d["n"] for d in per.values()) or 1
    shares = {cls: round(d["n"] / total, 4) for cls, d in per.items()}
    return {"per_class": per, "negative_class_share": shares,
            "n_negatives": total}


def _participant_split(rows, *, seed: int = SEED, frac: float = 0.5):
    """Stable participant-disjoint halves. Hash-based so the assignment
    does not move as the cohort grows (the datasets/splits discipline)."""
    tr, te = [], []
    for r in rows:
        pid = str(r.get("participant_id"))
        h = hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()
        (tr if int(h[:8], 16) / 0xFFFFFFFF < frac else te).append(r)
    return tr, te


def _matched_specificity(rows, scores, target_sens):
    """Specificity at the lowest threshold reaching `target_sens`.
    Returns (specificity, sensitivity, threshold, n_judged) or Nones
    when the target sensitivity is unreachable at any threshold."""
    y = np.asarray([1 if is_target(r) else 0 for r in rows], int)
    s = np.asarray([np.nan if v is None else float(v) for v in scores],
                   float)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return None, None, None, int(keep.sum())
    best = None
    for thr in np.unique(np.concatenate([s, [s.max() + 1e-9]])):
        pred = s >= thr
        sens = float(pred[y == 1].mean())
        spec = float((~pred[y == 0]).mean())
        if sens >= target_sens and (best is None or spec > best[0]):
            best = (spec, sens, float(thr))
    if best is None:
        return None, None, None, int(keep.sum())
    return best[0], best[1], best[2], int(keep.sum())


def flag_performance(rows, *, target_sensitivity: float = 0.80,
                     seed: int = SEED) -> dict:
    """F1 evidence: sensitivity for the 2:1 target, specificity against
    the whole negative battery, and the head-vs-B3 comparison at matched
    sensitivity that decides which rule ships."""
    # Specificity is the BATTERY's specificity, so its denominator is
    # the negative battery — non-2:1 flutter is neither the target nor
    # a negative, and counting it as one made the reported specificity
    # and n_negative_scans disagree about what they measured (review
    # finding).
    scored = [r for r in rows if is_target(r) or not is_flutter(r)]
    y = [1 if is_target(r) else 0 for r in scored]
    pid = [str(r.get("participant_id")) for r in scored]
    head = [head_flag(r) for r in scored]
    pred = [0 if v is None else int(v) for v in head]
    no_read = [v is None for v in head]
    res = evaluate_binary(y, pred, no_read_mask=no_read,
                          participant_ids=pid, n_boot=400, seed=seed)
    battery = hard_negative_battery(rows, rule=head_flag)
    out = {
        "n_flutter_scans": int(sum(1 for r in rows if is_flutter(r))),
        "n_target_scans": int(sum(y)),
        "n_negative_scans": battery["n_negatives"],
        "negative_class_share": battery["negative_class_share"],
        "per_negative_class": battery["per_class"],
        "sensitivity_2to1": (None if res.sensitivity != res.sensitivity
                             else round(res.sensitivity, 4)),
        "sensitivity_ci95": [round(x, 4) for x in res.sens_ci],
        "specificity_battery": (None if res.specificity != res.specificity
                                else round(res.specificity, 4)),
        "specificity_ci95": [round(x, 4) for x in res.spec_ci],
        "no_read_rate": round(res.no_read_rate, 4),
        "false_positives_per_1000_analysed":
            res.summary()["false_positives_per_1000_analysed"],
    }
    # baselines at their own operating points
    for name, rule in (("b1", b1_rate_only),
                       ("b2", b2_rate_and_naive_dispersion),
                       ("b3", b3_interpretable)):
        v = [rule(r) for r in scored]
        r_ = evaluate_binary(y, [0 if x is None else int(x) for x in v],
                             no_read_mask=[x is None for x in v],
                             participant_ids=pid, n_boot=200, seed=seed)
        out[f"{name}_sensitivity"] = (None if r_.sensitivity != r_.sensitivity
                                      else round(r_.sensitivity, 4))
        out[f"{name}_specificity"] = (None if r_.specificity != r_.specificity
                                      else round(r_.specificity, 4))
        out[f"{name}_no_read_rate"] = round(r_.no_read_rate, 4)
    # B4 (demographics + rate) is a fitted score, so it gets a
    # participant-disjoint split of its own and is judged on held-out
    # rows only — otherwise it would be scored on data it memorized
    tr, te = _participant_split(rows, seed=seed)
    b4_scores = b4_demographics_and_rate(tr, te) if (tr and te) else None
    if b4_scores is None:
        out["b4_specificity_at_matched_sens"] = None
        out["b4_reason"] = ("no participant-disjoint split with both "
                            "classes present in the training half")
    else:
        spec, sens, _, n_j = _matched_specificity(te, list(b4_scores),
                                                  target_sensitivity)
        out["b4_specificity_at_matched_sens"] = (None if spec is None
                                                 else round(spec, 4))
        out["b4_sensitivity_at_match"] = (None if sens is None
                                          else round(sens, 4))
        out["b4_n_test_rows"] = int(n_j)

    # ---- the decision comparison ------------------------------------
    # The head is a FIXED boolean rule, so it has one operating point;
    # sweeping a score for it would invent a curve it does not have.
    # B3 has a tunable knob, so B3 is matched TO the head — and on the
    # rows the head actually judged, or the head would "win" purely by
    # declining the rows B3 got wrong (review finding: it did).
    # the SAME denominator as specificity_battery above: targets plus
    # the negative battery, never non-2:1 flutter (which is neither).
    # And a participant-disjoint split, because B3's coupling knob has
    # to be CHOSEN somewhere: tuning it on the same rows it is scored on
    # let it find razor-thin in-sample splits the head (a fixed rule)
    # cannot answer (review finding).
    tr_s, te_s = _participant_split(scored, seed=seed)
    judged = [r for r in te_s if head_flag(r) is not None]
    h_sens, h_spec, h_n = _rates(judged, [head_flag(r) for r in judged])
    b_spec = b_sens = None
    if h_sens is not None and tr_s and judged:
        # pick B3's threshold on TRAIN to reach the head's sensitivity,
        # then apply that fixed threshold to the held-out rows
        _, _, thr, _ = _matched_specificity(
            tr_s, [_b3_score(r) for r in tr_s], h_sens)
        if thr is not None:
            b_sens, b_spec, _ = _rates(
                judged, [_b3_score(r) >= thr for r in judged])
    # ... and B3 unrestricted, so what the abstentions bought is visible
    b_all_spec, b_all_sens, _, b_all_n = _matched_specificity(
        scored, [_b3_score(r) for r in scored], target_sensitivity)
    out.update({
        "matched_sensitivity_target": target_sensitivity,
        "comparison_is_held_out": True,
        "head_specificity_at_matched_sens": (None if h_spec is None
                                             else round(h_spec, 4)),
        "b3_specificity_at_matched_sens": (None if b_spec is None
                                           else round(b_spec, 4)),
        "head_sensitivity_at_match": (None if h_sens is None
                                      else round(h_sens, 4)),
        "b3_sensitivity_at_match": (None if b_sens is None
                                    else round(b_sens, 4)),
        "comparison_rows": h_n,
        # B3 judging EVERYTHING, at the pre-registered target: the gap
        # between this and b3_specificity_at_matched_sens is exactly
        # what the head's abstention discipline is worth
        "b3_specificity_all_rows": (None if b_all_spec is None
                                    else round(b_all_spec, 4)),
        "b3_sensitivity_all_rows": (None if b_all_sens is None
                                    else round(b_all_sens, 4)),
        "b3_n_all_rows": b_all_n,
    })
    return out


def per_ratio_report(rows) -> dict:
    """F2: sensitivity broken out by conduction ratio, INCLUDING the
    ratios expected to be missed. A measured zero is the honest result;
    an absent number is a gate failure."""
    per = {}
    for r in rows:
        if not is_flutter(r):
            continue
        ratio = str(r.get("conduction_ratio") or "UNRECORDED")
        d = per.setdefault(ratio, {"n": 0, "n_flagged": 0,
                                   "n_abstained": 0})
        d["n"] += 1
        v = head_flag(r)
        if v is None:
            d["n_abstained"] += 1
        elif v:
            d["n_flagged"] += 1
    for ratio, d in per.items():
        judged = d["n"] - d["n_abstained"]
        d["sensitivity"] = (round(d["n_flagged"] / judged, 4)
                            if judged else None)
        d["sensitivity_ci95"] = ([round(x, 4) for x in
                                  wilson_ci(d["n_flagged"], judged)]
                                 if judged else None)
        d["abstention_rate"] = (round(d["n_abstained"] / d["n"], 4)
                                if d["n"] else None)
    slow = per.get("FOUR_TO_ONE") or {}
    return {"per_ratio": per,
            "slow_block_sensitivity": slow.get("sensitivity"),
            "slow_block_n": slow.get("n", 0),
            "note": ("4:1 flutter conducts at ~75 bpm and is expected to "
                     "be indistinguishable from normal sinus rhythm by "
                     "pulse timing — docs/flutter_limitations.md")}


def serial_signature(rows) -> dict:
    """F3: does the latent-atrial-rate fit separate flutter series from
    non-flutter series? Series shorter than the configured floor are
    reported and excluded, never scored as zero."""
    series = {}
    for r in rows:
        sid = r.get("series_id") or r.get("participant_id")
        series.setdefault(str(sid), []).append(r)
    scored, short, all_lengths = [], 0, []
    for sid, rs in sorted(series.items()):
        rates = [_f(_feat(x, "rate", "median_bpm")) for x in rs]
        rates = [x for x in rates if x is not None]
        all_lengths.append(len(rates))
        if len(rates) < 3:
            short += 1
            continue
        fit = latent_atrial_fit(rates)
        scored.append({"series_id": sid,
                       "n_scans": len(rates),
                       "is_flutter_series": bool(
                           any(is_flutter(x) for x in rs)),
                       "score": fit["score"],
                       "atrial_bpm": fit["atrial_bpm"]})
    y = np.asarray([1 if s["is_flutter_series"] else 0 for s in scored],
                   int)
    s = np.asarray([s["score"] for s in scored], float)
    auc = None
    if y.size and 0 < int(y.sum()) < y.size:
        from evaluation.afib_metrics import _auroc
        auc = round(float(_auroc(y, s)), 4)
    return {"series_auc": auc,
            "n_flutter_series": int(y.sum()) if y.size else 0,
            "n_nonflutter_series": int((1 - y).sum()) if y.size else 0,
            "n_series_too_short": short,
            # the SHORTEST series in the cohort, before the >= 3 filter:
            # reporting the post-filter minimum made the gate's own
            # floor unfailable (review finding)
            "min_scans_per_series": (min(all_lengths)
                                     if all_lengths else 0),
            "per_series": scored}


def fairness(rows, *, min_group_participants: int = 5,
             axes=("fitzpatrick_group",),
             darkest_bands=(5, 6)) -> dict:
    """F4: detection and coverage parity across subgroups, at
    PARTICIPANT level. A group below the floor is None-rated, which reds
    the gate by name rather than vanishing."""
    out = {}
    for axis in axes:
        groups = {}
        for r in rows:
            g = r.get(axis)
            if g is None:
                continue
            groups.setdefault(g, []).append(r)
        table, det, cov = {}, {}, {}
        for g, rs in groups.items():
            pids = {str(r.get("participant_id")) for r in rs}
            targets = [r for r in rs if is_target(r)]
            judged = [r for r in rs if head_flag(r) is not None]
            t_pids = {str(r.get("participant_id")) for r in targets}
            hit_pids = {str(r.get("participant_id")) for r in targets
                        if head_flag(r) is True}
            row = {"n_scans": len(rs), "n_participants": len(pids),
                   "n_target_participants": len(t_pids),
                   "coverage": (round(len(judged) / len(rs), 4)
                                if rs else None),
                   "detection_rate": (round(len(hit_pids) / len(t_pids),
                                            4) if t_pids else None)}
            if len(pids) < min_group_participants:
                row["rated"] = False
                row["reason"] = (f"{len(pids)} participants "
                                 f"(< {min_group_participants})")
            else:
                row["rated"] = True
                if row["detection_rate"] is not None:
                    det[g] = row["detection_rate"]
                if row["coverage"] is not None:
                    cov[g] = row["coverage"]
            table[str(g)] = row
        unrated = [g for g, r in table.items() if not r["rated"]]
        # a ratio over ONE group is 1.0 by construction: a monochrome
        # cohort must read as unrated parity, not perfect parity
        worst_det = (min(det.values()) / max(det.values())
                     if len(det) >= 2 and max(det.values()) > 0 else None)
        worst_cov = (min(cov.values()) / max(cov.values())
                     if len(cov) >= 2 and max(cov.values()) > 0 else None)
        out[axis] = {
            "groups": table,
            "detection_parity_ratio_worst": (
                None if (unrated or worst_det is None)
                else round(worst_det, 4)),
            "coverage_ratio_worst": (None if (unrated or worst_cov is None)
                                     else round(worst_cov, 4)),
            "unrated_groups": unrated,
            "n_rated_groups": len(det),
            "darkest_band_present": bool(
                any(g in groups for g in darkest_bands)),
        }
    return out


# ------------------------------------------ Task 6: combined endpoint
def combined_referral(afib_flag, flutter_flag) -> dict:
    """Fuse the two heads into ONE referral. Pure function of two head
    verdicts — no re-decision, no model, and no rhythm named.

    A None from either head means that head could not judge; the
    referral still fires if the OTHER head is positive, because a
    positive finding does not need unanimity. Both None is a no-read.
    """
    a = None if afib_flag is None else bool(afib_flag)
    f = None if flutter_flag is None else bool(flutter_flag)
    if a is None and f is None:
        return {"refer": None, "trigger": None,
                "sentence_key": None,
                "reason": "neither head could judge this scan"}
    if a:
        # the AF sentence already exists and is the stronger statement;
        # it is used verbatim, and the flag rides as internal evidence
        return {"refer": True, "trigger": ("both" if f else "irregular"),
                "sentence_key": "afib"}
    if f:
        # the trigger names the OBSERVATION, never a rhythm: "flutter"
        # here would have been a rhythm name in the output of the one
        # function whose contract is that it names none (review finding)
        return {"refer": True, "trigger": "regular_tachy",
                "sentence_key": "regular_tachy"}
    return {"refer": False, "trigger": None, "sentence_key": None}


def combined_endpoint(rows, *, prevalence=None, seed: int = SEED,
                      min_sensitivity: float = MIN_ENDPOINT_SENSITIVITY
                      ) -> dict:
    """Task 6: AF head alone vs flag alone vs combined, on the SAME rows,
    against the combined truth (AF or flutter present).

    Reported at deployment prevalence, because a cohort PPV computed on
    a cardioversion-clinic mix is not a shippable number.
    """
    prev = dict(prevalence or DEFAULT_PREVALENCE)
    combined_prev = float(prev.get("afib", 0.0)) + \
        float(prev.get("flutter", 0.0))
    pid = [str(r.get("participant_id")) for r in rows]

    def _truth(kind):
        if kind == "afib":
            return [1 if str(r.get("rhythm")) == "AFIB" else 0
                    for r in rows]
        if kind == "flutter":
            return [1 if is_flutter(r) else 0 for r in rows]
        return [1 if (str(r.get("rhythm")) == "AFIB" or is_flutter(r))
                else 0 for r in rows]

    out = {}
    candidates = {
        "afib_alone": ([r.get("afib_flag") for r in rows], "combined"),
        "flutter_alone": ([head_flag(r) for r in rows], "combined"),
        "combined": ([combined_referral(r.get("afib_flag"),
                                        head_flag(r))["refer"]
                      for r in rows], "combined"),
    }
    for name, (verdicts, truth_kind) in candidates.items():
        y = _truth(truth_kind)
        pred = [0 if v is None else int(v) for v in verdicts]
        nr = [v is None for v in verdicts]
        res = evaluate_binary(y, pred, no_read_mask=nr,
                              participant_ids=pid, n_boot=400, seed=seed)
        se, sp = res.sensitivity, res.specificity
        proj = (projected_ppv(se, sp, [combined_prev])[0]
                if se == se and sp == sp else None)
        out[name] = {
            "sensitivity": None if se != se else round(se, 4),
            "sensitivity_ci95": [round(x, 4) for x in res.sens_ci],
            "specificity": None if sp != sp else round(sp, 4),
            "specificity_ci95": [round(x, 4) for x in res.spec_ci],
            "no_read_rate": round(res.no_read_rate, 4),
            "n_analysed": res.n,
            "at_deployment_prevalence": (
                None if proj is None else {
                    "prevalence": round(combined_prev, 5),
                    "ppv": round(proj["ppv"], 4),
                    "npv": round(proj["npv"], 5),
                    "false_positives_per_1000_scans":
                        round(proj["false_positives_per_1000_scans"], 2),
                    "true_positives_per_1000_scans":
                        round(proj["true_positives_per_1000_scans"], 3),
                    "false_alarms_per_true_case":
                        (None if proj["false_alarms_per_true_case"] ==
                         float("inf")
                         else round(proj["false_alarms_per_true_case"],
                                    2))}),
        }
    # The ranking must not hand the win to OR-fusion by construction.
    # `combined` is `afib or flutter`, so its predictions dominate both
    # candidates elementwise and a sensitivity-first sort ALWAYS crowns
    # it — measured on the v0.6 cohort: combined won on sensitivity
    # (0.41 vs 0.33) while generating 2.9x the false alarms per true
    # case (10.2 vs 4.4). At screening prevalence that burden is the
    # deciding number, so candidates are ranked by it, and only among
    # those that actually catch enough to be worth referring on.
    eligible = [(n, d) for n, d in out.items()
                if d["sensitivity"] is not None
                and d["sensitivity"] >= min_sensitivity
                and (d["at_deployment_prevalence"] or {}).get(
                    "false_alarms_per_true_case") is not None]
    eligible.sort(key=lambda kv: (
        kv[1]["at_deployment_prevalence"]["false_alarms_per_true_case"],
        -kv[1]["sensitivity"]))
    recommended = eligible[0][0] if eligible else None
    return {"candidates": out,
            "prevalence_used": prev,
            "min_sensitivity_required": min_sensitivity,
            "ranked_by": "false_alarms_per_true_case at deployment "
                         "prevalence, among candidates meeting the "
                         "minimum sensitivity",
            "recommended": recommended,
            "no_recommendation_reason": (
                None if recommended else
                "no candidate reached the minimum combined-endpoint "
                f"sensitivity of {min_sensitivity} — on this cohort the "
                "endpoint decision cannot be made on the numbers"),
            "note": ("PPV is projected to deployment prevalence; the "
                     "cohort's own PPV is not shippable. Sensitivity is "
                     "against the COMBINED endpoint (AF or flutter), "
                     "which is what the referral claims.")}


# ------------------------------------------------------ the evaluation
class FlutterHarnessError(RuntimeError):
    pass


def rows_from_dataset(dataset_dir, *, heads_cfg=None) -> list:
    """Run the PRODUCTION path over a registered dataset and build one
    labeled row per recording.

    Respiration comes from the torso-motion second decode
    (rppg/respiration.py) — an independent physical channel, never the
    facial trace the intervals are built from, which would be circular.
    The flag is the REAL head's verdict, not a reimplementation of it.
    """
    from datasets.schema import Rhythm
    from heads import get_head
    from inference.pipeline import run_with_details
    from rppg._filters import moving_average_detrend
    from rppg.respiration import (respiratory_rate_from_motion,
                                  torso_motion_series)
    d = pathlib.Path(dataset_dir)
    mans = sorted(d.glob("*.recording.json"))
    if not mans:
        raise FlutterHarnessError(f"no *.recording.json manifests in {d}")
    head = get_head("flutter")
    rows = []
    for mp in mans:
        man = json.loads(mp.read_text())
        video = d / str(man.get("video_path") or "")
        if not video.exists():
            continue
        # the pipeline takes the capture block, not the manifest path:
        # it is the optics provenance (lock states, illuminance,
        # profile) that changes how the scan is read
        res, det = run_with_details(str(video),
                                    manifest=(man.get("capture") or {}))
        try:
            t, y, fps = torso_motion_series(str(video))
            rate, conc = respiratory_rate_from_motion(t, y, fps)
            resp = {"t": t,
                    "y": moving_average_detrend(y, int(12.0 * fps)),
                    "rate_brpm": rate, "quality": conc}
        except (OSError, IOError, ValueError):
            resp = None
        anns = man.get("rhythm_annotations") or [{}]
        ann = anns[0]
        # participant-level metadata (skin-tone group, age, sex) is
        # Participant data and rides in its own sidecar; without it the
        # fairness gate has nothing to rate
        pp = d / f"{mp.name.split('.recording.json')[0]}.participant.json"
        part = json.loads(pp.read_text()) if pp.exists() else {}
        ctx = {"scan_outcome": res.outcome.value, "cfg": heads_cfg or {},
               "respiration": resp,
               "age_years": part.get("age_years") or
               (man.get("participant_context") or {}).get("age")}
        hr = head.run(det["lattice"], ctx)
        from features.flutter import flutter_features

        class _R:
            outcome = res.outcome
        feats = flutter_features(_R(), det["lattice"], cfg=heads_cfg or {},
                                 respiration=resp,
                                 age_years=ctx["age_years"])
        rows.append({
            "recording_id": man.get("recording_id"),
            "participant_id": man.get("participant_id"),
            "session_id": man.get("session_id"),
            "series_id": man.get("participant_id"),
            # NOT the dataset version — that is constant across a
            # dataset and would collapse the whole hard-negative
            # battery into one class. A finer negative class is an
            # optional per-recording annotation; otherwise the
            # adjudicated rhythm is the class.
            "cohort": (man.get("participant_context") or {})
            .get("negative_class"),
            "rhythm": ann.get("rhythm") or Rhythm.UNINTERPRETABLE.value,
            "conduction_ratio": ann.get("conduction_ratio"),
            "ecg_rate_bpm": ann.get("ventricular_rate_bpm"),
            "fitzpatrick_group": part.get("fitzpatrick_group"),
            "age_years": ctx["age_years"],
            "sex": part.get("sex"),
            "fps": det["lattice"].fps,
            "features": feats if feats.get("available") else {},
            "head_flag": hr.value.get("regular_tachy_flag"),
            "afib_flag": (res.predicted_class == "AFIB_SUGGESTIVE"
                          if res.outcome.value == "ACCEPT" else None),
            "scan_outcome": res.outcome.value,
        })
    return rows


def evaluate_flutter_dataset(rows, *, signal_domain, runs_root=None,
                             gates_path=None,
                             production_path: bool = False,
                             prevalence=None, seed: int = SEED) -> dict:
    """Run every §F evaluation over labeled rows and record the result.

    `signal_domain` is REQUIRED and `production_path` defaults to False:
    a permissive default would let hand-built or surrogate rows be
    recorded as production facial-rPPG evidence, which is precisely the
    evidence class that can open a gate. The caller must state what the
    rows are.

    Refuses an empty cohort rather than writing a scoreboard entry full
    of Nones that would read like a completed evaluation.
    """
    rows = list(rows or [])
    if not rows:
        raise ValueError("no labeled scans supplied — an evaluation "
                         "with no cohort is not an evaluation")
    gcfg = load_flutter_gates(gates_path)
    t1 = gcfg.get("f1_flag_performance") or {}
    t4 = gcfg.get("f4_fairness") or {}
    target = float(t1.get("min_sensitivity_2to1", 0.80))

    f0 = rate_accuracy(rows)
    f1 = flag_performance(rows, target_sensitivity=target, seed=seed)
    f2 = per_ratio_report(rows)
    f3 = serial_signature(rows)
    f4 = fairness(rows,
                  min_group_participants=int(
                      t4.get("min_group_participants", 5)))
    endpoint = combined_endpoint(rows, prevalence=prevalence, seed=seed)
    f5 = {"combined_endpoint_decision": endpoint.get("recommended") or "",
          "combined_endpoint": endpoint}

    pids = sorted({str(r.get("participant_id")) for r in rows})
    sids = sorted({str(r.get("session_id") or r.get("recording_id"))
                   for r in rows})
    evidence = {
        "data": {"signal_domain": signal_domain,
                 # computed, never asserted: disjointness holds only when
                 # the cohort actually contains more than one of each
                 "participant_disjoint": len(pids) > 1,
                 "session_disjoint": len(sids) > 1,
                 "production_path": bool(production_path),
                 "n_scans": len(rows), "n_participants": len(pids)},
        "f0": f0, "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
    }
    from evaluation.flutter_gates import evaluate_flutter_gates
    verdict = evaluate_flutter_gates(gcfg, evidence)
    # the id must distinguish EVIDENCE CLASSES, not just cohorts: a
    # facial_rppg re-run of the same recordings would otherwise land in
    # the same run dir and silently replace the synthetic run's
    # gate_results.json — the file _promote_flutter reads and hashes
    # (review finding)
    run_id = "flut-" + hashlib.sha256(
        json.dumps({"n": len(rows),
                    "ids": sorted(str(r.get("recording_id"))
                                  for r in rows),
                    "domain": signal_domain,
                    "production_path": bool(production_path),
                    "gates_version": gcfg.get("gates_version")},
                   sort_keys=True).encode()).hexdigest()[:12]
    root = pathlib.Path(runs_root or DEFAULT_RUNS)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rule = shipping_rule(evidence, gates_path=gates_path)
    (run_dir / "gate_results.json").write_text(json.dumps(verdict,
                                                          indent=1))
    (run_dir / "flutter_record.json").write_text(json.dumps(
        {"run_id": run_id, "n_scans": len(rows),
         "n_participants": len(pids), "shipping_rule": rule,
         "gates_version": gcfg.get("gates_version")}, indent=1))
    (run_dir / "model.json").write_text(json.dumps(
        {"kind": "regular_tachy_pattern_rule",
         "shipping_rule": rule,
         "target": "TWO_TO_ONE",
         "criteria": ["sustained_band", "below_dispersion_floor",
                      "respiratory_modulation_absent"]}, indent=1))
    entry = {"kind": "evaluation", "run_id": run_id,
             "gates_version": gcfg.get("gates_version"),
             "evidence": evidence, "verdict": {
                 "all_gates_green": verdict["all_gates_green"],
                 "promotion_open": verdict["promotion_open"]},
             "shipping_rule": rule}
    append_scoreboard(entry, runs_root=runs_root)
    return {"run_id": run_id, "run_dir": str(run_dir),
            "evidence": evidence, "verdict": verdict,
            "shipping_rule": rule, "combined_endpoint": endpoint}
