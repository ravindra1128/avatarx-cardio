"""
Canonical AvatarX AFib dataset schema.

Design constraints that drove this schema:

1.  REGULATORY TRACEABILITY. Every record must be traceable to a code commit,
    a dataset version, a capture configuration and an adjudication decision.
    IEC 62304 and any future FDA submission require this; retrofitting it is
    far more expensive than carrying it from record #1.

2.  PARTICIPANT IS THE UNIT OF ANALYSIS. `participant_id` is the only
    identifier that may be used for train/val/test partitioning. Session and
    window identifiers exist for bookkeeping, never for splitting.

3.  CAPTURE PROVENANCE IS EVIDENCE. Prior AvatarX research established that
    lossy compression destroys the pulse signal irrecoverably (deep rPPG
    models fail at CRF >= 22 and training on compressed data does not
    recover it). Codec/CRF is therefore a first-class validity field, not
    metadata: a record captured above the CRF ceiling is not a weak record,
    it is an invalid one.

4.  RHYTHM IS AN INTERVAL PROPERTY, NOT A RECORDING PROPERTY. A 90 s scan
    can contain sinus rhythm, three PACs and a run of SVT. Recording-level
    labels destroy exactly the information a rhythm model needs, so labels
    are stored as time-bounded annotations and the recording-level label is
    a derived convenience field.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields, asdict
from enum import Enum
from typing import Optional, List, Dict, Any
import json
import hashlib


# --------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------

class Rhythm(str, Enum):
    """Rhythm annotation vocabulary.

    Deliberately NOT a binary AF/non-AF field. The hard negatives are the
    whole point: a model validated only against clean sinus rhythm has an
    unmeasured specificity. Prior AvatarX research found that in the only
    facial-video study reporting per-arrhythmia error, sinus arrhythmia
    produced a 20% false-positive rate and PVCs 6.7%.
    """
    SINUS = "SINUS"
    SINUS_BRADYCARDIA = "SINUS_BRADYCARDIA"
    SINUS_TACHYCARDIA = "SINUS_TACHYCARDIA"
    RESPIRATORY_SINUS_ARRHYTHMIA = "RESPIRATORY_SINUS_ARRHYTHMIA"
    AFIB = "AFIB"
    ATRIAL_FLUTTER = "ATRIAL_FLUTTER"
    PAC = "PAC"
    PAC_FREQUENT = "PAC_FREQUENT"
    PVC = "PVC"
    PVC_FREQUENT = "PVC_FREQUENT"
    SVT = "SVT"
    PACED = "PACED"
    OTHER = "OTHER"
    UNINTERPRETABLE = "UNINTERPRETABLE"

    @classmethod
    def hard_negatives(cls) -> set["Rhythm"]:
        """Rhythms that an interval-based detector is most likely to call AF."""
        return {cls.PAC, cls.PAC_FREQUENT, cls.PVC, cls.PVC_FREQUENT,
                cls.ATRIAL_FLUTTER, cls.SVT, cls.RESPIRATORY_SINUS_ARRHYTHMIA}


class AFibPattern(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PAROXYSMAL = "PAROXYSMAL"
    PERSISTENT = "PERSISTENT"
    LONGSTANDING_PERSISTENT = "LONGSTANDING_PERSISTENT"
    PERMANENT = "PERMANENT"
    UNKNOWN = "UNKNOWN"


class ConductionRatio(str, Enum):
    """v0.6: AV conduction of an atrial tachyarrhythmia — the label field
    that decides whether the arrhythmia is visible to a camera at all.

    Flutter's atria fire ~250-300/min and the AV node passes a fraction.
    At 2:1 the pulse is ~150 and metronomic; at 4:1 it is ~75 and
    metronomic — i.e. indistinguishable from resting sinus rhythm by
    pulse timing (docs/flutter_limitations.md, invariant F-c). Recording
    the ratio is therefore not bookkeeping: it is what makes the known
    misses countable instead of assertable.
    """
    TWO_TO_ONE = "TWO_TO_ONE"
    THREE_TO_ONE = "THREE_TO_ONE"
    FOUR_TO_ONE = "FOUR_TO_ONE"
    VARIABLE = "VARIABLE"

    @property
    def divisor(self):
        """Atrial rate / this = ventricular rate. None for VARIABLE —
        variable block has no single divisor, and pretending otherwise
        would manufacture an atrial-rate estimate the data cannot carry."""
        return {"TWO_TO_ONE": 2.0, "THREE_TO_ONE": 3.0,
                "FOUR_TO_ONE": 4.0}.get(self.value)


class FlutterType(str, Enum):
    """Typical (cavotricuspid-isthmus-dependent) vs atypical. Naming
    either one requires F-wave morphology on a 12-lead ECG; the facial
    pulse cannot distinguish them and never will."""
    TYPICAL = "TYPICAL"
    ATYPICAL = "ATYPICAL"
    UNKNOWN = "UNKNOWN"


class MonkTone(int, Enum):
    """Monk Skin Tone scale, 1 (lightest) to 10 (darkest).

    Monk is used rather than Fitzpatrick because FDA's recent cuffless-BP
    draft guidance specifies Monk banding (1-4 / 5-7 / 8-10), and because
    Fitzpatrick is a sun-reactivity scale, not a reflectance scale.
    Fitzpatrick is retained as a secondary field for literature comparison.
    """
    MST_1 = 1; MST_2 = 2; MST_3 = 3; MST_4 = 4; MST_5 = 5
    MST_6 = 6; MST_7 = 7; MST_8 = 8; MST_9 = 9; MST_10 = 10

    @property
    def band(self) -> str:
        if self.value <= 4:  return "1-4"
        if self.value <= 7:  return "5-7"
        return "8-10"


class SyncMethod(str, Enum):
    """How video and ECG clocks were aligned. Governs `sync_uncertainty_ms`."""
    HARDWARE_TRIGGER = "HARDWARE_TRIGGER"      # shared trigger line; sub-ms
    LED_FLASH_MARKER = "LED_FLASH_MARKER"      # optical marker in frame + ECG event
    NTP_SOFTWARE = "NTP_SOFTWARE"              # clock sync only; tens of ms
    CROSS_CORRELATION = "CROSS_CORRELATION"    # post-hoc signal alignment
    NONE = "NONE"                              # unsynchronised -- beat-level use PROHIBITED


class MeasurementClass(str, Enum):
    """v0.2 invariant 9: every surfaced quantity carries its epistemic
    class. app/ may render MEASURED and INFERRED_RHYTHM always;
    INFERRED_FITNESS (v0.4) may be rendered ONLY while the §V block of
    configs/gates.yaml is green AND clinically signed — until then a
    fitness inference exists solely inside gated heads and evaluation
    artifacts. RESEARCH_SYNTHETIC exists solely for the quarantined
    falsification lab and must never reach a consumer artifact.
    RESEARCH_VASCULAR (v0.4 arterial-stiffness track) is the
    head_vascular class: research-flagged, and NEVER renderable on any
    user-facing surface while any gate of the vascular block of
    configs/gates.yaml is red or its signoff is unsigned (V-a).
    RESEARCH_RHYTHM (v0.6 atrial-flutter track) is the head_flutter
    class: a rhythm-family research output — regular-tachyarrhythmia
    pattern evidence — never renderable while any gate of the flutter
    block is red or unsigned (F-b). It is deliberately NOT
    INFERRED_RHYTHM: that class is renderable by app/ on sight, and an
    ungated flutter output must be unrenderable by construction, not by
    remembering to check."""
    MEASURED = "MEASURED"                  # pulse waveform, beat times, rate, SQI
    INFERRED_RHYTHM = "INFERRED_RHYTHM"    # rhythm class, AF suggestion, stars
    INFERRED_FITNESS = "INFERRED_FITNESS"  # v0.4 fitness category/trend — §V-gated
    RESEARCH_SYNTHETIC = "RESEARCH_SYNTHETIC"  # falsification-lab artifacts
    RESEARCH_VASCULAR = "RESEARCH_VASCULAR"    # stiffness research head — gated
    RESEARCH_RHYTHM = "RESEARCH_RHYTHM"        # v0.6 flutter research head — §F


def public_head_results(rows) -> list:
    """Invariant 9 at the app serialization boundary: RESEARCH_* head
    results never ride in a consumer payload — the head-level inert
    branch is the first layer, this filter is the second (a
    config-enabled research head must not reach the browser JSON even
    as a stub)."""
    out = []
    for r in rows or []:
        cls = str((r or {}).get("measurement_class", ""))
        if not cls.startswith("RESEARCH_"):
            out.append(r)
    return out


class ScanOutcome(str, Enum):
    ACCEPT = "ACCEPT"
    REPEAT_SCAN = "REPEAT_SCAN"
    NO_RESULT = "NO_RESULT"


class Split(str, Enum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    INTERNAL_TEST = "INTERNAL_TEST"     # locked; opened once
    EXTERNAL_TEST = "EXTERNAL_TEST"     # separate site, separate personnel
    UNASSIGNED = "UNASSIGNED"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass
class RhythmAnnotation:
    """A time-bounded rhythm label on the ECG timeline.

    `t_start_s` / `t_end_s` are seconds from ECG recording start. Conversion
    to the video timeline uses SyncRecord.offset_ms and is the ONLY sanctioned
    way to place a label on video.
    """
    t_start_s: float
    t_end_s: float
    rhythm: Rhythm
    annotator_id: str
    confidence: Optional[float] = None            # annotator's own 0-1
    adjudicated: bool = False                     # True if >=2 readers agreed
    adjudication_note: Optional[str] = None
    ventricular_rate_bpm: Optional[float] = None
    ectopy_burden_per_min: Optional[float] = None  # for PAC/PVC arms
    # v0.6 flutter block — additive, None on every pre-v0.6 annotation.
    # A flutter POSITIVE label is only a label when all of these are
    # present and EP-adjudicated on a 12-lead: the pulse cannot name the
    # rhythm, so the label carries the entire naming burden.
    atrial_rate_bpm: Optional[float] = None
    conduction_ratio: Optional[ConductionRatio] = None
    flutter_type: Optional[FlutterType] = None
    adjudicator_id: Optional[str] = None          # EP-level reader
    adjudication_leads: Optional[int] = None      # leads the reader saw

    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s


# v0.6: the plausible atrial-rate window for a flutter label. Typical
# flutter runs 250-300/min; the band is widened for atypical and
# rate-controlled circuits, but a value outside it is a transcription
# error, not a rare patient.
FLUTTER_ATRIAL_RATE_BPM = (180.0, 450.0)
FLUTTER_MIN_ADJUDICATION_LEADS = 12


def flutter_label_problems(a: "RhythmAnnotation", *,
                           ecg_leads: Optional[int] = None) -> list:
    """Why this annotation is not a usable ATRIAL_FLUTTER positive.

    Empty list = usable. The bar is deliberately high and fails closed:
    flutter cannot be named from a pulse, and single-lead ECG cannot
    reliably name it either (F-wave morphology needs the inferior leads).
    A recording whose ECG is single-lead remains a perfectly good
    rate/regularity comparator and a perfectly good NEGATIVE — this
    function governs only the positive label.
    """
    if a.rhythm is not Rhythm.ATRIAL_FLUTTER:
        return ["not an ATRIAL_FLUTTER annotation"]
    bad = []
    if not a.adjudicated:
        bad.append("not adjudicated")
    if not str(a.adjudicator_id or "").strip():
        bad.append("no adjudicator identity recorded")
    leads = a.adjudication_leads if a.adjudication_leads is not None \
        else ecg_leads
    if leads is None:
        bad.append("adjudication lead count not recorded")
    elif int(leads) < FLUTTER_MIN_ADJUDICATION_LEADS:
        bad.append(f"adjudicated on {int(leads)} lead(s) — a flutter "
                   f"positive needs {FLUTTER_MIN_ADJUDICATION_LEADS}-lead "
                   "ECG with EP-level reading (F-wave morphology)")
    if a.conduction_ratio is None:
        bad.append("no conduction ratio recorded — the per-ratio gate "
                   "(F2) and the known-miss registry (F-c) cannot be "
                   "computed without it")
    if a.flutter_type is None:
        bad.append("no flutter type recorded")
    r = a.atrial_rate_bpm
    lo, hi = FLUTTER_ATRIAL_RATE_BPM
    if r is None:
        bad.append("no atrial rate recorded")
    else:
        try:
            r = float(r)
        except (TypeError, ValueError):
            bad.append("atrial rate is not a number")
            r = None
        if r is not None and not (lo <= r <= hi):
            bad.append(f"atrial rate {r:.0f}/min outside the plausible "
                       f"window [{lo:.0f}, {hi:.0f}]")
    return bad


@dataclass
class CaptureConfig:
    """Everything about the optical path. Validity, not metadata.

    `is_valid_for_beat_analysis()` is the gate that keeps unusable recordings
    out of beat-level training and evaluation.
    """
    phone_model: str
    os_version: str
    camera: str                                   # "front" | "rear"
    width: int
    height: int
    nominal_fps: float
    measured_fps_mean: Optional[float] = None
    measured_fps_jitter_ms: Optional[float] = None   # SD of frame period
    codec: str = "unknown"                        # "ffv1" | "h264" | "hevc" | ...
    crf: Optional[int] = None                     # None => lossless
    bitrate_mbps: Optional[float] = None
    exposure_locked: bool = False
    awb_locked: bool = False
    gain_locked: bool = False
    beautification_disabled: bool = True
    ois_eis_state: str = "unknown"                # "off" | "ois" | "eis" | "both"
    illuminance_lux_mean: Optional[float] = None
    illuminance_lux_sd: Optional[float] = None
    colour_temperature_k: Optional[float] = None
    distance_cm: Optional[float] = None
    mount: str = "unknown"                        # "handheld" | "tripod" | "stand"

    # Ceilings established by prior AvatarX research.
    CRF_CEILING = 18
    # Bits-per-pixel-per-frame floor for lossy capture. Phone HARDWARE encoders
    # expose bitrate, not CRF, so a CRF-only gate is unevaluable exactly where
    # it matters. 0.12 bpp ~ x264 CRF 18 at 1080p; 1080p60 needs >= ~15 Mbps.
    BPP_FLOOR = 0.12
    LUX_FLOOR = 100.0
    FPS_FLOOR_BEAT = 30.0

    def is_lossless(self) -> bool:
        return self.codec.lower() in {"ffv1", "huffyuv", "rawvideo", "none"}

    def bits_per_pixel_per_frame(self) -> Optional[float]:
        """Bitrate normalised by spatial and temporal resolution.

        The comparable quantity across codecs/encoders when CRF is unknown.
        None if bitrate was not recorded.
        """
        if self.bitrate_mbps is None:
            return None
        fps = self.measured_fps_mean or self.nominal_fps
        if not fps or not self.width or not self.height:
            return None
        return self.bitrate_mbps * 1e6 / (self.width * self.height * fps)

    def is_valid_for_beat_analysis(self) -> tuple[bool, list[str]]:
        """Hard validity gate. Returns (ok, reasons_for_failure).

        Lossy capture passes via EITHER an explicit CRF <= 18 (software encode,
        research rig) OR bits-per-pixel-per-frame >= 0.12 (hardware encode,
        phones). Lossy capture with neither recorded is invalid — unknown
        compression is indistinguishable from fatal compression.
        """
        bad: list[str] = []
        if not self.is_lossless():
            crf_ok = self.crf is not None and self.crf <= self.CRF_CEILING
            bpp = self.bits_per_pixel_per_frame()
            bpp_ok = bpp is not None and bpp >= self.BPP_FLOOR
            if not (crf_ok or bpp_ok):
                if self.crf is None and bpp is None:
                    bad.append("lossy codec with neither CRF nor bitrate recorded")
                elif self.crf is not None and self.crf > self.CRF_CEILING:
                    bad.append(f"CRF {self.crf} > ceiling {self.CRF_CEILING}"
                               + (f" and bpp {bpp:.3f} < floor {self.BPP_FLOOR}"
                                  if bpp is not None else ""))
                else:
                    bad.append(f"bitrate {self.bitrate_mbps} Mbps -> bpp "
                               f"{bpp:.3f} < floor {self.BPP_FLOOR}")
        if (self.measured_fps_mean or self.nominal_fps) < self.FPS_FLOOR_BEAT:
            bad.append(f"fps below {self.FPS_FLOOR_BEAT}")
        if self.illuminance_lux_mean is not None and \
           self.illuminance_lux_mean < self.LUX_FLOOR:
            bad.append(f"illuminance {self.illuminance_lux_mean:.0f} lux "
                       f"< floor {self.LUX_FLOOR:.0f}")
        if not (self.exposure_locked and self.awb_locked):
            bad.append("auto-exposure and/or auto-white-balance not locked")
        if not self.beautification_disabled:
            bad.append("beautification/filter pipeline active")
        return (len(bad) == 0), bad


@dataclass
class SyncRecord:
    """Video/ECG clock alignment and its uncertainty.

    `sync_uncertainty_ms` propagates directly into every beat-level metric.
    If it is not materially smaller than the IBI error being measured, the
    measurement is of the synchronisation, not of the algorithm.

    PHYSICAL LIMIT (v2 correction): a SINGLE optical flash localises the sync
    event only to within one frame — worst case ±half a frame period
    (±16.7 ms at 30 fps, ±8.3 ms at 60 fps; SD = frame/sqrt(12) = 9.6 / 4.8 ms).
    A single flash therefore CANNOT meet the 5 ms budget at consumer frame
    rates. A pseudo-random sequence of K flashes, cross-correlated against the
    ECG event channel, reaches ~frame/sqrt(12·K) (K=20 at 60 fps -> ~1.1 ms)
    and additionally measures clock drift. LED sync is accordingly valid only
    with `n_marker_events >= 10`.
    """
    method: SyncMethod
    offset_ms: float                    # video_t = ecg_t + offset_ms/1000
    sync_uncertainty_ms: float
    drift_ppm: Optional[float] = None
    verified_at_end: bool = False       # re-checked at recording end
    n_marker_events: int = 1            # optical marker events used for alignment

    BEAT_LEVEL_MAX_UNCERTAINTY_MS = 5.0
    MIN_MARKER_EVENTS = 10

    def is_valid_for_beat_analysis(self) -> tuple[bool, list[str]]:
        bad: list[str] = []
        if self.method is SyncMethod.NONE:
            bad.append("no synchronisation performed")
        if self.method is SyncMethod.LED_FLASH_MARKER and \
                self.n_marker_events < self.MIN_MARKER_EVENTS:
            bad.append(
                f"only {self.n_marker_events} optical marker event(s): a single "
                f"flash is bounded by ±half a frame (≥8 ms at 60 fps); "
                f">= {self.MIN_MARKER_EVENTS} events required for the 5 ms budget")
        if self.method is SyncMethod.CROSS_CORRELATION:
            bad.append("post-hoc cross-correlation aligns using the signal being "
                       "measured (circular); not valid for beat-level evaluation")
        if self.sync_uncertainty_ms > self.BEAT_LEVEL_MAX_UNCERTAINTY_MS:
            bad.append(f"sync uncertainty {self.sync_uncertainty_ms:.1f} ms "
                       f"> {self.BEAT_LEVEL_MAX_UNCERTAINTY_MS} ms")
        if not self.verified_at_end:
            bad.append("synchronisation not re-verified at recording end")
        return (len(bad) == 0), bad


@dataclass
class Participant:
    participant_id: str
    site_id: str
    age_years: Optional[int] = None
    sex: Optional[str] = None                     # "M" | "F" | "other"
    monk_tone: Optional[MonkTone] = None
    fitzpatrick: Optional[int] = None
    self_reported_ethnicity: Optional[str] = None
    bmi: Optional[float] = None
    facial_hair: Optional[str] = None             # "none" | "light" | "heavy"
    glasses: bool = False
    makeup: Optional[str] = None
    # Clinical
    known_afib: Optional[bool] = None
    afib_pattern: AFibPattern = AFibPattern.NOT_APPLICABLE
    hypertension: Optional[bool] = None
    diabetes: Optional[bool] = None
    heart_failure: Optional[bool] = None
    prior_stroke: Optional[bool] = None
    cha2ds2_vasc: Optional[int] = None
    medications: List[str] = field(default_factory=list)
    on_rate_control: Optional[bool] = None        # beta blocker / CCB / digoxin
    on_anticoagulant: Optional[bool] = None
    resting_hr_bpm: Optional[float] = None
    # Governance
    consent_version: Optional[str] = None
    consent_video_retention: Optional[bool] = None
    enrolled_date: Optional[str] = None


@dataclass
class Recording:
    """One face-scan session with its simultaneous ECG."""
    recording_id: str
    participant_id: str
    session_id: str
    site_id: str

    video_path: str
    ecg_path: str
    video_start_utc: str                          # ISO 8601 with timezone
    video_end_utc: str
    ecg_start_utc: str
    ecg_end_utc: str
    duration_s: float

    capture: CaptureConfig
    sync: SyncRecord

    ecg_sampling_hz: float = 500.0
    ecg_leads: int = 1
    ecg_device: str = "unknown"

    # Optional simultaneous CONTACT PPG (finger/ear) — collection-rig only,
    # never part of the consumer path. It disambiguates the central ambiguity
    # of Gate 1b: a beat present on ECG but absent in facial rPPG is either an
    # extraction failure (present in contact PPG) or a true pulse deficit
    # (absent in both). Without this channel the two are indistinguishable and
    # Gate 1b penalises physiology. Deficit rate is itself an AF feature.
    contact_ppg_path: Optional[str] = None
    contact_ppg_hz: Optional[float] = None

    rhythm_annotations: List[RhythmAnnotation] = field(default_factory=list)
    ecg_rpeaks_s: Optional[List[float]] = None    # reference beat times, ECG clock

    protocol_id: str = "unknown"                  # controlled vs real-world
    lighting_condition: str = "unknown"
    motion_condition: str = "stationary"          # "stationary"|"natural"|"deliberate"
    operator: str = "unknown"                     # "technician" | "self"

    split: Split = Split.UNASSIGNED
    dataset_version: str = "unversioned"
    excluded: bool = False
    exclusion_reason: Optional[str] = None
    # v0.4 additive blocks (None for rhythm-track recordings): the
    # standardized activity actually performed, and the user-entered
    # workload/reference context. Types are defined below in the v0.4
    # section; loaders coerce them (datasets/io.py).
    challenge: Optional["ChallengeRecord"] = None
    participant_context: Optional["ParticipantContext"] = None

    # ---------------------------------------------------------------- helpers
    def recording_level_rhythm(self) -> Rhythm:
        """Derived convenience label. NEVER the training target for a rhythm
        model -- it is for cohort tabulation only.

        Precedence reflects clinical salience, not duration: any adjudicated
        AF makes the recording an AF recording.
        """
        if not self.rhythm_annotations:
            return Rhythm.UNINTERPRETABLE
        present = {a.rhythm for a in self.rhythm_annotations}
        for r in (Rhythm.AFIB, Rhythm.ATRIAL_FLUTTER, Rhythm.SVT,
                  Rhythm.PVC_FREQUENT, Rhythm.PAC_FREQUENT,
                  Rhythm.PVC, Rhythm.PAC):
            if r in present:
                return r
        # else the longest-duration annotation
        return max(self.rhythm_annotations, key=lambda a: a.duration_s()).rhythm

    def flutter_label_problems(self) -> list:
        """v0.6: why this recording cannot serve as a flutter POSITIVE.
        Empty list means it can. A recording with no flutter annotation
        reports that plainly rather than silently passing — a cohort
        builder must never read "no problems" as "confirmed flutter"."""
        anns = [a for a in self.rhythm_annotations
                if a.rhythm is Rhythm.ATRIAL_FLUTTER]
        if not anns:
            return ["no ATRIAL_FLUTTER annotation on this recording"]
        best, fewest = None, None
        for i, a in enumerate(anns):
            bad = flutter_label_problems(a, ecg_leads=self.ecg_leads)
            if fewest is None or len(bad) < fewest:
                best, fewest = [f"annotation {i}: {x}" for x in bad], len(bad)
            if not bad:
                return []
        return best or []

    def afib_seconds(self) -> float:
        return sum(a.duration_s() for a in self.rhythm_annotations
                   if a.rhythm is Rhythm.AFIB)

    def afib_fraction(self) -> float:
        return self.afib_seconds() / self.duration_s if self.duration_s else 0.0

    def is_valid_for_beat_analysis(self) -> tuple[bool, list[str]]:
        ok_c, bad_c = self.capture.is_valid_for_beat_analysis()
        ok_s, bad_s = self.sync.is_valid_for_beat_analysis()
        bad = bad_c + bad_s
        if self.ecg_rpeaks_s is None:
            bad.append("no reference R-peaks")
        if self.ecg_sampling_hz < 250:
            bad.append(f"ECG {self.ecg_sampling_hz} Hz below 250 Hz")
        if not any(a.adjudicated for a in self.rhythm_annotations):
            bad.append("no adjudicated rhythm annotation")
        return (len(bad) == 0), bad

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def content_hash(self) -> str:
        """Stable hash for provenance. Excludes mutable split assignment."""
        d = asdict(self)
        d.pop("split", None)
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]


# v0.2 (M1.3, spec B.15): the ONLY sanctioned sentences for rate flags.
# Heads never compose text; app/ may render these keys verbatim only.
RATE_FLAG_SENTENCES = {
    "brady": ("Your pulse was slower than typical during this scan "
              "(under 50 beats per minute). If you feel faint, unusually "
              "tired, or this repeats, mention it to a clinician."),
    "tachy": ("Your pulse stayed above 100 beats per minute during this "
              "scan. If you feel unwell or this repeats, contact a "
              "clinician."),
}

# v0.6 (spec B.22, invariant F-a): the ONLY sanctioned sentence for the
# regular-tachyarrhythmia pattern flag, and it may render only once the
# §F gates are green AND signed (evaluation.flutter_gates.
# flutter_render_allowed). It names NO rhythm, by design and by
# physiology: a metronomic pulse at ~150 is produced by 2:1 flutter, by
# SVT/AVNRT and by sinus tachycardia alike, and separating them needs
# F waves on an ECG — which facial video does not carry. The sentence
# routes to the one thing that CAN name it.
REGULAR_TACHY_SENTENCES = {
    "regular_tachy": ("A sustained fast, unusually regular pulse was "
                      "detected. This can occur with several heart-rhythm "
                      "conditions — please have an ECG."),
}

# v0.7 (spec B.23, invariant G-d): the ONLY sanctioned sentence for the
# regularity head's irregular finding, renderable only once the §R gates
# are green AND signed (evaluation.regularity_gates.
# regularity_render_allowed). It names NO rhythm — not AFib, not ectopy —
# and it names the benign explanation first, because the head's dominant
# false positive is a healthy person breathing. Escalation to the
# AFib-suggestive sentence remains head_afib's job; this head never
# escalates on its own.
REGULARITY_SENTENCES = {
    "irregular": ("Your pulse rhythm looked irregular during this scan. "
                  "This is often benign (for example, normal "
                  "breathing-related variation), but if it repeats or you "
                  "have symptoms, an ECG can tell you why."),
}


@dataclass
class ScanResult:
    """What the inference pipeline returns. Deliberately not a bare label."""
    recording_id: str
    outcome: ScanOutcome
    afib_probability: Optional[float] = None      # calibrated; None if NO_RESULT
    predicted_class: Optional[str] = None         # SINUS|AFIB_SUGGESTIVE|OTHER_IRREGULAR|HIGH_RATE
    signal_quality_index: Optional[float] = None
    usable_beats: Optional[int] = None
    analysed_seconds: Optional[float] = None
    mean_pulse_rate_bpm: Optional[float] = None
    no_read_reasons: List[str] = field(default_factory=list)
    # Confidence rating (v0.1.5) -- the user-visible expression of the
    # evidence gates; a result below 3 stars can never carry
    # predicted_class == "AFIB_SUGGESTIVE" (enforced in decision logic)
    confidence_stars: Optional[int] = None            # 1..5
    confidence_limiting_factor: Optional[str] = None  # lowest-margin check
    # v0.2 (schema v2): per-head results + capture metadata. Each entry in
    # head_results is a HeadResult.to_dict() record carrying its own
    # measurement_class; capture_meta is device/capture bookkeeping.
    schema_version: int = 2
    head_results: List[dict] = field(default_factory=list)
    capture_meta: Optional[dict] = None
    # Provenance -- required for regulatory traceability
    model_version: str = "unknown"
    code_commit: str = "unknown"
    calibration_version: str = "unknown"
    config_hash: str = "unknown"

    # v0.2 invariant 9: EVERY field is classified. MEASURED covers
    # recorded facts of the scan event (identifiers, capture provenance,
    # signal measurements); INFERRED_RHYTHM covers everything the decision
    # layer concluded. NOTHING in ScanResult may ever be
    # RESEARCH_SYNTHETIC — that class exists only inside the quarantined
    # falsification lab. head_results is a MEASURED container whose items
    # carry their own authoritative per-item class.
    FIELD_CLASSES = {
        "recording_id": MeasurementClass.MEASURED,
        "outcome": MeasurementClass.INFERRED_RHYTHM,
        "afib_probability": MeasurementClass.INFERRED_RHYTHM,
        "predicted_class": MeasurementClass.INFERRED_RHYTHM,
        "signal_quality_index": MeasurementClass.MEASURED,
        "usable_beats": MeasurementClass.MEASURED,
        "analysed_seconds": MeasurementClass.MEASURED,
        "mean_pulse_rate_bpm": MeasurementClass.MEASURED,
        "no_read_reasons": MeasurementClass.INFERRED_RHYTHM,
        "confidence_stars": MeasurementClass.INFERRED_RHYTHM,
        "confidence_limiting_factor": MeasurementClass.INFERRED_RHYTHM,
        "schema_version": MeasurementClass.MEASURED,
        "head_results": MeasurementClass.MEASURED,
        "capture_meta": MeasurementClass.MEASURED,
        "model_version": MeasurementClass.MEASURED,
        "code_commit": MeasurementClass.MEASURED,
        "calibration_version": MeasurementClass.MEASURED,
        "config_hash": MeasurementClass.MEASURED,
    }

    # Fixed, audited hint per limiting factor for the low-star sentence.
    # This table is part of the sanctioned-text surface: only these
    # strings can ever be interpolated, never free text.
    _STAR_HINTS = {
        "face": "Position your face inside the oval.",
        "framing": "Position your face inside the oval.",
        "lighting": "Improve lighting: face a bright, even light.",
        "exposure": "Hold still and wait for the camera to settle.",
        "motion": "Hold still.",
        "tracking": "Hold still.",
        "frame_rate": "Close other apps using the camera.",
        "frame_rate_floor": "Close other apps using the camera.",
        "timestamps": "Close other apps using the camera.",
        "signal_snr": "More light on your face, with no hair over the "
                      "forehead.",
        "cross_roi_coherence": "Even light on both cheeks and forehead.",
        "beat_timing": "Hold still in bright, even light.",
        "prelim_beats": "Hold still in bright, even light.",
        "sqi": "Hold still in bright, even light.",
    }

    def user_facing_text(self) -> str:
        """The ONLY sanctioned user-facing strings.

        No variant of this method may return a diagnosis. 21 CFR 870.2790
        states verbatim that PPG analysis software "is not intended to
        provide a diagnosis".
        """
        suffix = (f" Confidence: {int(self.confidence_stars)} of 5."
                  if self.confidence_stars is not None else "")
        if self.outcome is not ScanOutcome.ACCEPT:
            if self.confidence_stars is not None and self.confidence_stars < 3:
                hint = self._STAR_HINTS.get(
                    self.confidence_limiting_factor or "",
                    "Please try again in better light, holding still.")
                return ("We could see your pulse, but not clearly enough to "
                        f"check your rhythm this time. {hint}" + suffix)
            return ("We couldn't read your pulse clearly enough this time. "
                    "Please try again in better light, holding still." + suffix)
        if self.predicted_class == "AFIB_SUGGESTIVE":
            return ("We detected an irregular rhythm that can be associated with "
                    "atrial fibrillation. This is not a diagnosis. Please share "
                    "this result with a clinician, who may recommend an ECG."
                    + suffix)
        if self.predicted_class == "OTHER_IRREGULAR":
            return ("We detected some irregular beats. This is common and often "
                    "harmless, but if it recurs, mention it to a clinician."
                    + suffix)
        if self.predicted_class == "HIGH_RATE":
            return ("Your pulse was faster than usual during this scan. "
                    "If you feel unwell or this repeats, contact a clinician."
                    + suffix)
        return ("No irregular rhythm was detected during this scan. "
                "This scan cannot rule out atrial fibrillation, which often "
                "comes and goes." + suffix)


def scan_result_from_dict(d: dict) -> ScanResult:
    """Read a stored ScanResult of ANY schema version (v0.2 migration
    shim). v1 records (no schema_version) gain empty v2 fields; unknown
    future versions fail closed."""
    d = dict(d)
    ver = int(d.get("schema_version", 1))
    if ver > 2:
        raise ValueError(f"ScanResult schema_version {ver} is newer than "
                         "this build (reads <= 2)")
    known = {f for f in ScanResult.__dataclass_fields__}
    extra = {k for k in d if k not in known}
    if extra:
        raise ValueError(f"unknown ScanResult fields {sorted(extra)}")
    d["schema_version"] = 2
    d.setdefault("head_results", [])
    d.setdefault("capture_meta", None)
    if not isinstance(d.get("outcome"), ScanOutcome):
        d["outcome"] = ScanOutcome(d["outcome"])
    return ScanResult(**d)


# ===========================================================================
# v0.4 — cardiorespiratory recovery + fitness track (additive schema rev).
# ScanResult stays at schema_version 2 (no field changes); the rev adds the
# session-level objects below, the challenge / participant-context manifest
# blocks, the CPET label block, and MeasurementClass.INFERRED_FITNESS with
# its §V-gated rendering invariant.
# ===========================================================================

@dataclass
class ChallengeRecord:
    """The standardized activity actually performed (v0.4). Prescription
    comes from protocol/challenges.py; achieved values from the activity
    verifier — the camera verifies WORKLOAD here, never physiology."""
    protocol_id: str
    cadence_prescribed: float                  # reps or steps per minute
    cadence_achieved: Optional[float] = None
    reps: Optional[int] = None
    transition_s: Optional[float] = None       # activity end -> still & framed


@dataclass
class Medications:
    """Rate-limiting/rate-modifying medication answers (self-reported).
    beta_blocker/ccb/ivabradine make HR-based fitness inference
    unreliable: decision logic hard-routes head_fitness to trend-only."""
    beta_blocker: bool = False
    ccb: bool = False                          # rate-limiting Ca-channel blocker
    ivabradine: bool = False
    stimulant: bool = False
    thyroid: bool = False

    @property
    def rate_limiting(self) -> bool:
        return bool(self.beta_blocker or self.ccb or self.ivabradine)


@dataclass
class ParticipantContext:
    """User-entered context for workload + reference values. Weight and
    height are MEASURED-BY-THE-USER inputs — never inferred from the
    face (v0.4 hard rule)."""
    age: Optional[int] = None
    sex: Optional[str] = None                  # "female" | "male" | "other"
    measured_weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    meds: Medications = field(default_factory=Medications)
    activity_ipaq: Optional[str] = None        # IPAQ-SF category self-report


@dataclass
class CpetLabel:
    """Reference cardiopulmonary exercise test label (v0.4) — the ONLY
    sanctioned VO2 ground truth. Training/evaluation only; a VO2 number
    NEVER renders on any user-facing surface in any version."""
    vo2peak_mlkgmin: float
    modality: str                              # treadmill | cycle | ...
    protocol: str                              # e.g. "bruce", "ramp_20w"
    rer_peak: Optional[float] = None
    hr_peak: Optional[float] = None
    effort_criteria: List[str] = field(default_factory=list)
    avg_window_s: Optional[float] = None
    cart: Optional[str] = None                 # metabolic cart model
    lab: Optional[str] = None
    test_date: Optional[str] = None


def cpet_from_dict(d: dict) -> CpetLabel:
    """Fail-closed CPET label reader for <id>.cpet.json."""
    if not isinstance(d, dict):
        raise ValueError("cpet label must be an object")
    known = {f.name for f in dataclass_fields(CpetLabel)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"cpet label: unknown fields {sorted(unknown)}")
    for k in ("vo2peak_mlkgmin", "modality", "protocol"):
        if k not in d:
            raise ValueError(f"cpet label missing required field {k!r}")
    v = float(d["vo2peak_mlkgmin"])
    if not (5.0 <= v <= 95.0):
        raise ValueError(f"cpet vo2peak {v} mL/kg/min outside the "
                         "plausible range [5, 95] — fail closed")
    if d.get("rer_peak") is not None and not (0.7 <= float(d["rer_peak"])
                                              <= 1.6):
        raise ValueError("cpet rer_peak outside [0.7, 1.6]")
    if d.get("hr_peak") is not None and not (60.0 <= float(d["hr_peak"])
                                             <= 230.0):
        raise ValueError("cpet hr_peak outside [60, 230]")
    ec = d.get("effort_criteria", [])
    if not isinstance(ec, list) or not all(isinstance(x, str) for x in ec):
        raise ValueError("cpet effort_criteria must be a list of strings")
    return CpetLabel(**dict(d, vo2peak_mlkgmin=v))


@dataclass
class VascularReference:
    """Reference arterial-stiffness measurement for one session (v0.4
    vascular track) — the ONLY sanctioned stiffness ground truth
    (carotid-femoral PWV by tonometry, SphygmoCor/Complior class).
    Training/evaluation only; no stiffness quantity ever renders on a
    user-facing surface while the vascular gates are red (V-a/V-b)."""
    cfpwv_mps: float                            # carotid-femoral PWV, m/s
    device_model: str                           # tonometry device + version
    operator: str
    cfpwv_mps_repeat: Optional[float] = None    # 2nd read (retest protocol)
    cavi: Optional[float] = None                # optional CAVI
    brachial_sbp_mmhg: Optional[float] = None   # same-visit brachial BP
    brachial_dbp_mmhg: Optional[float] = None
    hr_at_measurement_bpm: Optional[float] = None
    meds_antihypertensive: bool = False
    meds_vasoactive: bool = False
    # visit-recorded demographics: BASELINE-ONLY inputs (T2 defense) —
    # they may never enter the vascular head model, only B1-B5
    age_years: Optional[float] = None
    sex: Optional[str] = None                   # "female"|"male"|"other"
    fitzpatrick_group: Optional[int] = None     # 1..6, operator-recorded
    ambient_temp_note: Optional[str] = None
    skin_temp_note: Optional[str] = None
    site_id: Optional[str] = None
    device_label: Optional[str] = None          # capture-rig label (strata)
    test_date: Optional[str] = None


def _plaus(name: str, v, lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    if not (lo <= v <= hi):
        raise ValueError(f"vascular reference {name} {v} outside the "
                         f"plausible range [{lo}, {hi}] — fail closed")
    return v


def vascular_from_dict(d: dict) -> VascularReference:
    """Fail-closed stiffness reference reader for <id>.pwv.json."""
    if not isinstance(d, dict):
        raise ValueError("vascular reference must be an object")
    known = {f.name for f in dataclass_fields(VascularReference)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(
            f"vascular reference: unknown fields {sorted(unknown)}")
    for k in ("cfpwv_mps", "device_model", "operator"):
        if k not in d:
            raise ValueError(f"vascular reference missing required "
                             f"field {k!r}")
    for k in ("device_model", "operator"):
        if not isinstance(d[k], str) or not d[k].strip():
            raise ValueError(f"vascular reference {k} must be a "
                             "non-empty string")
    out = dict(d)
    out["cfpwv_mps"] = _plaus("cfpwv_mps", d["cfpwv_mps"], 3.0, 25.0)
    out["cfpwv_mps_repeat"] = _plaus("cfpwv_mps_repeat",
                                     d.get("cfpwv_mps_repeat"), 3.0, 25.0)
    out["cavi"] = _plaus("cavi", d.get("cavi"), 2.0, 20.0)
    sbp = _plaus("brachial_sbp_mmhg", d.get("brachial_sbp_mmhg"),
                 70.0, 260.0)
    dbp = _plaus("brachial_dbp_mmhg", d.get("brachial_dbp_mmhg"),
                 35.0, 150.0)
    if sbp is not None and dbp is not None and sbp <= dbp:
        raise ValueError("vascular reference systolic pressure must "
                         "exceed diastolic — fail closed")
    out["brachial_sbp_mmhg"], out["brachial_dbp_mmhg"] = sbp, dbp
    out["hr_at_measurement_bpm"] = _plaus(
        "hr_at_measurement_bpm", d.get("hr_at_measurement_bpm"),
        30.0, 220.0)
    for k in ("meds_antihypertensive", "meds_vasoactive"):
        v = d.get(k, False)
        if not isinstance(v, bool):
            raise ValueError(f"vascular reference {k} must be a JSON "
                             f"boolean, got {v!r} — a string 'no' would "
                             "ingest as truthy, fail closed")
        out[k] = v
    for k in ("site_id", "device_label", "ambient_temp_note",
              "skin_temp_note", "test_date"):
        v = d.get(k)
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError(f"vascular reference {k} must be None or a "
                             "non-empty string")
    fg = d.get("fitzpatrick_group")
    if fg is not None:
        if isinstance(fg, bool) or not isinstance(fg, (int, float)) \
                or float(fg) != int(fg):
            raise ValueError("vascular reference fitzpatrick_group must "
                             "be an integer 1..6")
        fg = int(fg)
        if not (1 <= fg <= 6):
            raise ValueError("vascular reference fitzpatrick_group must "
                             "be 1..6")
    out["fitzpatrick_group"] = fg
    out["age_years"] = _plaus("age_years", d.get("age_years"),
                              18.0, 100.0)
    sx = d.get("sex")
    if sx is not None and sx not in ("female", "male", "other"):
        raise ValueError("vascular reference sex must be "
                         "female|male|other")
    return VascularReference(**out)


@dataclass
class ContactPpg:
    """Synchronized contact-PPG reference waveform sidecar (<id>.ppg.json,
    v0.4 vascular track). Collection-rig only — the fidelity study's
    contact arm; never a consumer input. Sample clock: sample i sits at
    t0_video_s + i / fs_hz on the video capture clock."""
    fs_hz: float
    samples: List[float]
    t0_video_s: float = 0.0
    device: str = "unknown"


def contact_ppg_from_dict(d: dict) -> ContactPpg:
    """Fail-closed contact-PPG reader; a reference that cannot support
    beat segmentation is rejected, not degraded."""
    if not isinstance(d, dict):
        raise ValueError("contact ppg must be an object")
    known = {f.name for f in dataclass_fields(ContactPpg)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"contact ppg: unknown fields {sorted(unknown)}")
    for k in ("fs_hz", "samples"):
        if k not in d:
            raise ValueError(f"contact ppg missing required field {k!r}")
    fs = float(d["fs_hz"])
    if not (25.0 <= fs <= 10000.0):
        raise ValueError(f"contact ppg fs_hz {fs} outside [25, 10000] — "
                         "morphology needs beat-resolving bandwidth")
    samples = d["samples"]
    if not isinstance(samples, list) or len(samples) < int(2 * fs):
        raise ValueError("contact ppg shorter than 2 s — cannot segment "
                         "beats, fail closed")
    vals = [float(x) for x in samples]
    if not all(v == v and abs(v) != float("inf") for v in vals):
        raise ValueError("contact ppg contains non-finite samples")
    t0 = float(d.get("t0_video_s", 0.0))
    if not (t0 == t0 and abs(t0) != float("inf")):
        raise ValueError("contact ppg t0_video_s must be finite — it "
                         "anchors the sample clock, fail closed")
    return ContactPpg(fs_hz=fs, samples=vals, t0_video_s=t0,
                      device=str(d.get("device", "unknown")))


# ---------------------------------------------------------------------------
# v0.5 vasotone track — provocation protocol + perfusion-index reference
# ---------------------------------------------------------------------------

# The pre-registered maneuver vocabulary (docs/vasotone_track.md). The two
# null arms are first-class maneuvers: null_optics is the T1 defense
# (physiology at rest, rig optics deliberately varied), null_rest defines
# the natural within-session drift distribution.
MANEUVERS = ("cold_pressor", "paced_breathing", "mental_arithmetic",
             "posture_change", "null_optics", "null_rest")
_PHASES = ("baseline", "stimulus", "recovery")


@dataclass
class ProvocationRecord:
    """One provocation session's protocol record (v0.5 vasotone,
    <id>.provocation.json). Phase marks are on the video capture clock;
    the capture rig writes them from its own timed prompts. Vascular
    tone is DYNAMIC: everything downstream is a within-session
    baseline-vs-response comparison, never an absolute level (W-c)."""
    maneuver: str
    phase_marks: dict                       # phase -> (t0_s, t1_s)
    intensity: Optional[int] = None         # graded provocations, 1..3
    optics_log: Optional[str] = None        # REQUIRED for null_optics
    fitzpatrick_group: Optional[int] = None  # operator-recorded, 1..6 —
                                             # STRATA ONLY, never a model
                                             # input (W4 parity tables)
    notes: Optional[str] = None


def provocation_from_dict(d: dict) -> ProvocationRecord:
    """Fail-closed provocation reader."""
    if not isinstance(d, dict):
        raise ValueError("provocation record must be an object")
    known = {f.name for f in dataclass_fields(ProvocationRecord)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"provocation record: unknown fields "
                         f"{sorted(unknown)}")
    man = d.get("maneuver")
    if man not in MANEUVERS:
        raise ValueError(f"provocation maneuver {man!r} is not in the "
                         f"pre-registered set {list(MANEUVERS)}")
    pm = d.get("phase_marks")
    if not isinstance(pm, dict):
        raise ValueError("provocation phase_marks must be an object")
    marks: dict = {}
    for phase in ("baseline", "stimulus"):
        if phase not in pm:
            raise ValueError(f"provocation phase_marks missing "
                             f"{phase!r} — a response without its "
                             "baseline window is meaningless")
    for phase, span in pm.items():
        if phase not in _PHASES:
            raise ValueError(f"provocation phase {phase!r} unknown "
                             f"(sanctioned: {list(_PHASES)})")
        if (not isinstance(span, (list, tuple)) or len(span) != 2):
            raise ValueError(f"phase {phase} span must be [t0, t1]")
        t0, t1 = float(span[0]), float(span[1])
        if not (t0 == t0 and t1 == t1) or t0 < 0 or t1 <= t0:
            raise ValueError(f"phase {phase} span must be ordered "
                             "non-negative seconds")
        marks[phase] = (t0, t1)
    spans = sorted(marks.values())
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        if b0 < a1:
            raise ValueError("provocation phases overlap — the rig's "
                             "timed prompts must partition the session")
    inten = d.get("intensity")
    if inten is not None:
        if isinstance(inten, bool) or not isinstance(inten, int) or \
                not (1 <= inten <= 3):
            raise ValueError("provocation intensity must be an integer "
                             "1..3")
    log = d.get("optics_log")
    if man == "null_optics" and (not isinstance(log, str)
                                 or not log.strip()):
        raise ValueError("null_optics arm requires a recorded "
                         "optics_log — an undocumented perturbation "
                         "cannot anchor gate W1, fail closed")
    fg = d.get("fitzpatrick_group")
    if fg is not None:
        if isinstance(fg, bool) or not isinstance(fg, (int, float)) \
                or float(fg) != int(fg) or not (1 <= int(fg) <= 6):
            raise ValueError("provocation fitzpatrick_group must be an "
                             "integer 1..6")
        fg = int(fg)
    return ProvocationRecord(maneuver=man, phase_marks=marks,
                             intensity=inten, optics_log=log,
                             fitzpatrick_group=fg,
                             notes=d.get("notes"))


@dataclass
class PiTrace:
    """Contact pulse-oximeter perfusion-index trace (v0.5 vasotone,
    <id>.pi.json) — the reference arm for gate W0, recorded on a hand
    NOT involved in the maneuver. Sample i sits at t0_video_s + i/fs_hz
    on the video capture clock."""
    fs_hz: float
    values: List[float]
    t0_video_s: float = 0.0
    device: str = "unknown"


def pi_from_dict(d: dict) -> PiTrace:
    """Fail-closed PI-trace reader."""
    if not isinstance(d, dict):
        raise ValueError("pi trace must be an object")
    known = {f.name for f in dataclass_fields(PiTrace)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"pi trace: unknown fields {sorted(unknown)}")
    for k in ("fs_hz", "values"):
        if k not in d:
            raise ValueError(f"pi trace missing required field {k!r}")
    fs = float(d["fs_hz"])
    if not (0.2 <= fs <= 1000.0):
        raise ValueError(f"pi trace fs_hz {fs} outside [0.2, 1000] — "
                         "PI is a slow trace but must resolve a "
                         "provocation phase")
    vals = d["values"]
    if not isinstance(vals, list) or len(vals) < int(30 * fs):
        raise ValueError("pi trace shorter than 30 s — cannot span a "
                         "baseline + response window, fail closed")
    out = [float(x) for x in vals]
    if not all(v == v and abs(v) != float("inf") for v in out):
        raise ValueError("pi trace contains non-finite values")
    if any(v <= 0 for v in out):
        raise ValueError("pi trace values must be positive "
                         "(percent units)")
    t0 = float(d.get("t0_video_s", 0.0))
    if not (t0 == t0 and abs(t0) != float("inf")):
        raise ValueError("pi trace t0_video_s must be finite")
    return PiTrace(fs_hz=fs, values=out, t0_video_s=t0,
                   device=str(d.get("device", "unknown")))


# The ONLY sanctioned session-level sentences (v0.4). Same audit contract
# as the rhythm tables: heads and app/ never compose fitness/session text;
# they render these keys verbatim. No sentence names a disease, claims a
# diagnosis, or contains a VO2 value.
SESSION_SENTENCES = {
    "recovery_complete": (
        "Your session is complete. These are your measured heart-rate "
        "recovery values — measurements, not a fitness rating."),
    "resting_only": (
        "Here are your resting measurements. Fitness-related metrics "
        "require the short guided activity, which this session did not "
        "include."),
    "repeat_protocol": (
        "The activity didn't match the target pace closely enough for "
        "recovery metrics to be comparable. Your vital signs are shown; "
        "please try the guided activity again."),
    "no_result_protocol": (
        "We couldn't verify the activity for this session, so recovery "
        "metrics aren't available. Your resting measurements are shown."),
    "safety_blocked": (
        "Based on your answers, please skip the activity portion and "
        "speak with a clinician before exercise testing. A resting scan "
        "is still available."),
}


# The ONLY sanctioned fitness-category and trend sentences (v0.4).
# UNRENDERABLE while any §V gate is red or unsigned — they exist so that
# IF §V ever opens, the surface still renders audited keys verbatim
# rather than composing text in app/ (invariant: sanctioned tables only).
# No sentence contains a number, a percentile, or a disease name.
FITNESS_CATEGORY_SENTENCES = {
    "below_typical": ("Your heart-rate recovery was below the typical "
                      "range for your age group in this session. This is "
                      "a wellness observation, not a medical finding."),
    "typical": ("Your heart-rate recovery was in the typical range for "
                "your age group in this session. This is a wellness "
                "observation, not a medical finding."),
    "above_typical": ("Your heart-rate recovery was above the typical "
                      "range for your age group in this session. This is "
                      "a wellness observation, not a medical finding."),
}
TREND_DIRECTION_SENTENCES = {
    "improving": ("Across your recent sessions, your heart-rate recovery "
                  "trend is improving."),
    "stable": ("Across your recent sessions, your heart-rate recovery "
               "trend is stable."),
    "declining": ("Across your recent sessions, your heart-rate recovery "
                  "trend is declining. If this continues, consider "
                  "mentioning it to a clinician."),
}


@dataclass
class SessionResult:
    """What `cli.py session` returns (v0.4): the three-phase protocol
    outcome. Every field is classified (invariant 9); the two
    INFERRED_FITNESS fields stay None on every surface while any §V gate
    is red or the clinical signoff is missing."""
    session_id: str
    outcome: ScanOutcome                       # fitness-metric availability
    protocol_id: Optional[str] = None
    activity_performed: bool = False
    safety_blocked: bool = False
    phases: dict = field(default_factory=dict)     # per-phase summaries
    compliance: Optional[dict] = None              # reps/cadence/transition
    hr_rest_bpm: Optional[float] = None
    rr_rest_brpm: Optional[float] = None
    hr_end_proxy_bpm: Optional[float] = None
    hrr30_bpm: Optional[float] = None
    hrr60_bpm: Optional[float] = None
    hrr120_bpm: Optional[float] = None
    recovery_slope_bpm_min: Optional[float] = None
    confidence_stars: Optional[int] = None
    confidence_limiting_factor: Optional[str] = None
    no_read_reasons: List[str] = field(default_factory=list)
    fitness_category: Optional[str] = None     # INFERRED_FITNESS — §V-gated
    trend: Optional[dict] = None               # INFERRED_FITNESS — §V-gated
    head_results: List[dict] = field(default_factory=list)
    capture_meta: Optional[dict] = None
    activity_tracker: Optional[str] = None     # cadence-verifier provenance
    schema_version: int = 1
    model_version: str = "unknown"
    code_commit: str = "unknown"
    calibration_version: str = "unknown"
    config_hash: str = "unknown"

    FIELD_CLASSES = {
        "session_id": MeasurementClass.MEASURED,
        "outcome": MeasurementClass.MEASURED,
        "protocol_id": MeasurementClass.MEASURED,
        "activity_performed": MeasurementClass.MEASURED,
        "safety_blocked": MeasurementClass.MEASURED,
        "phases": MeasurementClass.MEASURED,
        "compliance": MeasurementClass.MEASURED,
        "hr_rest_bpm": MeasurementClass.MEASURED,
        "rr_rest_brpm": MeasurementClass.MEASURED,
        "hr_end_proxy_bpm": MeasurementClass.MEASURED,
        "hrr30_bpm": MeasurementClass.MEASURED,
        "hrr60_bpm": MeasurementClass.MEASURED,
        "hrr120_bpm": MeasurementClass.MEASURED,
        "recovery_slope_bpm_min": MeasurementClass.MEASURED,
        "confidence_stars": MeasurementClass.MEASURED,
        "confidence_limiting_factor": MeasurementClass.MEASURED,
        "no_read_reasons": MeasurementClass.MEASURED,
        "fitness_category": MeasurementClass.INFERRED_FITNESS,
        "trend": MeasurementClass.INFERRED_FITNESS,
        "head_results": MeasurementClass.MEASURED,
        "capture_meta": MeasurementClass.MEASURED,
        "activity_tracker": MeasurementClass.MEASURED,
        "schema_version": MeasurementClass.MEASURED,
        "model_version": MeasurementClass.MEASURED,
        "code_commit": MeasurementClass.MEASURED,
        "calibration_version": MeasurementClass.MEASURED,
        "config_hash": MeasurementClass.MEASURED,
    }

    def user_facing_text(self) -> str:
        """Session-level sanctioned sentence — chosen by state, verbatim
        from SESSION_SENTENCES, with the standard confidence suffix."""
        suffix = ""
        if self.confidence_stars is not None:
            suffix = f" Confidence: {int(self.confidence_stars)} of 5."
        if self.safety_blocked:
            return SESSION_SENTENCES["safety_blocked"]
        if not self.activity_performed:
            return SESSION_SENTENCES["resting_only"] + suffix
        if self.outcome is ScanOutcome.NO_RESULT:
            return SESSION_SENTENCES["no_result_protocol"] + suffix
        if self.outcome is ScanOutcome.REPEAT_SCAN:
            return SESSION_SENTENCES["repeat_protocol"] + suffix
        return SESSION_SENTENCES["recovery_complete"] + suffix


def session_result_from_dict(d: dict) -> SessionResult:
    """Migration shim for SessionResult (reads schema_version <= 1;
    unknown fields and future versions fail closed)."""
    d = dict(d)
    ver = int(d.get("schema_version", 1))
    if ver > 1:
        raise ValueError(f"SessionResult schema_version {ver} is newer "
                         "than this build (reads <= 1)")
    known = {f.name for f in dataclass_fields(SessionResult)}
    extra = set(d) - known
    if extra:
        raise ValueError(f"unknown SessionResult fields {sorted(extra)}")
    d["schema_version"] = 1
    if not isinstance(d.get("outcome"), ScanOutcome):
        d["outcome"] = ScanOutcome(d["outcome"])
    return SessionResult(**d)
