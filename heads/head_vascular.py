"""head_vascular — arterial-stiffness research head (v0.4 vascular).

RESEARCH-FLAGGED (invariant V-a): while any gate of the `vascular:`
block of configs/gates.yaml is red or its signoff is unsigned, this head
produces NO user-facing output of any kind — no number, score, trend, or
color. On any surface the production pipeline can reach it returns an
inert, number-free result; the real estimate exists ONLY when a research
surface (cli.py process --heads vascular, or the evaluation harness)
supplies precomputed inputs via the context.

Quarantine posture: this module deliberately contains NO import of
research/* (the transitive walker audits every import in every reachable
module, function-scoped included). The research side computes features
and loads the model artifact, then hands both in via ctx; this head only
applies a plain linear artifact — dict arithmetic, no research code.

T2 invariant: the artifact may contain ONLY V0-surviving morphology
features. Demographics (age/sex/BP) live in baselines, never here.
"""
from __future__ import annotations

from datasets.schema import MeasurementClass
from heads.base import EndpointHead, HeadResult, register_head

# Must equal research.vascular.WATERMARK (pinned by test — the literal
# is duplicated here so the head never imports the quarantined package).
WATERMARK = ("VASCULAR MORPHOLOGY — RESEARCH ARTIFACT — "
             "NOT A MEASUREMENT")

_DEMOGRAPHIC_TOKENS = ("age", "sex", "sbp", "dbp", "bp", "weight",
                       "height", "bmi")


class VascularHead(EndpointHead):
    name = "vascular"
    version = "0.1.0-research"
    research_only = True
    required_inputs = ()

    def run(self, lattice, context: dict) -> HeadResult:
        ctx = context or {}
        if ctx.get("vascular_research_surface") is not True:
            # every non-research surface, including any accidental
            # config-enable in the production pipeline: inert and
            # number-free — nothing here can leak to a user
            return HeadResult(
                head=self.name, version=self.version,
                measurement_class=MeasurementClass.RESEARCH_VASCULAR,
                value={"available": False},
                confidence=None,
                reasons=["research-only head (V-a): outputs exist only "
                         "on research surfaces while the vascular gates "
                         "are red — nothing renders to users"])
        reasons = []
        feats = (ctx.get("vascular_features") or {})
        model = ctx.get("vascular_model") or {}
        surviving = list(ctx.get("surviving_features") or [])
        if not feats.get("available"):
            reasons.append("no morphology features for this scan (V-c "
                           "or quality refusal)")
        if not model.get("coef"):
            reasons.append("no trained vascular model on record — run "
                           "evaluate-vascular on a referenced dataset")
        if not surviving:
            reasons.append("no V0-surviving features — the fidelity "
                           "gate blocks all estimation (fail closed)")
        used = list(model.get("features") or [])
        for f in used:
            if any(tok == f or f.startswith(tok + "_")
                   for tok in _DEMOGRAPHIC_TOKENS):
                reasons.append(f"artifact contains demographic feature "
                               f"{f!r} — forbidden inside the model "
                               "(T2 defense), refusing")
        if model.get("coef") and not (
                len(used) == len(model.get("mu") or [])
                == len(model.get("sd") or [])
                == len(model.get("coef") or []) > 0):
            reasons.append("model artifact arrays disagree in length "
                           "(features/mu/sd/coef) — corrupt artifact, "
                           "refusing")
        if not reasons and any(f not in surviving for f in used):
            reasons.append("model artifact uses features outside the "
                           "V0-surviving set — stale artifact, refusing")
        fvals = (feats.get("features") or {})
        if not reasons and any(fvals.get(f) is None for f in used):
            missing = [f for f in used if fvals.get(f) is None]
            reasons.append(f"scan lacks required features {missing}")
        if reasons:
            return HeadResult(
                head=self.name, version=self.version,
                measurement_class=MeasurementClass.RESEARCH_VASCULAR,
                value={"available": False, "watermark": WATERMARK},
                confidence=None, reasons=reasons)
        z = [(float(fvals[f]) - float(mu)) / float(sd)
             for f, mu, sd in zip(used, model["mu"], model["sd"])]
        est = float(model["intercept"]) + sum(
            zi * float(c) for zi, c in zip(z, model["coef"]))
        rsd = float(model.get("residual_sd") or 0.0)
        ci = [round(est - 1.96 * rsd, 2), round(est + 1.96 * rsd, 2)]
        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.RESEARCH_VASCULAR,
            value={"available": True, "watermark": WATERMARK,
                   "estimate_cfpwv_mps": round(est, 2),
                   "ci95_mps": ci,
                   "ci_basis": "train-residual spread — naive, "
                               "pre-validation",
                   "model_kind": model.get("kind"),
                   "model_n_train": model.get("n_train"),
                   "features_used": used,
                   "feature_quality": feats.get("quality") or {},
                   "n_beats_used": feats.get("n_beats_used")},
            confidence=None,
            reasons=["research artifact — never a measurement, never "
                     "user-facing while the vascular gates are red "
                     "(V-a)"])


register_head(VascularHead)
