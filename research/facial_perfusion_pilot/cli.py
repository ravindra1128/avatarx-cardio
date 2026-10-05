"""FPR pilot CLI (isolated; does not touch the repo's top-level cli.py).

    python -m research.facial_perfusion_pilot.cli b7
    python -m research.facial_perfusion_pilot.cli b0 --bit-depth 8 --roi 150
    python -m research.facial_perfusion_pilot.cli init  <session_dir>
    python -m research.facial_perfusion_pilot.cli qualify <session_dir>
    python -m research.facial_perfusion_pilot.cli diagnostics <out_dir>
"""
from __future__ import annotations

import argparse
import json
import sys

from . import WATERMARK
from .contract import new_session, FprSession
from .stages import run_B7, quantisation_headroom
from . import qualify as _qualify


def _cmd_b7(_args) -> int:
    res = run_B7()
    print(WATERMARK + "\n")
    print(f"B7 {res.status}: {res.reason}")
    print(json.dumps(res.metrics, indent=2, default=float))
    return 0 if res.status == "PASS" else 1


def _cmd_b0(args) -> int:
    m = quantisation_headroom(args.bit_depth, args.roi * args.roi,
                              args.expected)
    passed = m["headroom_x"] >= m["headroom_min"]
    print(WATERMARK + "\n")
    print(f"B0 {'PASS' if passed else 'FAIL'}: {m['headroom_x']:.1f}x headroom "
          f"(floor {m['headroom_min']:.0f}x) for a {args.bit_depth}-bit sensor, "
          f"{args.roi}x{args.roi}px ROI, {args.expected*100:.2f}% AC/DC")
    print(json.dumps(m, indent=2, default=float))
    return 0 if passed else 1


def _cmd_init(args) -> int:
    s = new_session(args.session, manifest={"created_by": "fpr.cli"})
    print(f"created FPR session skeleton at {s.root}")
    miss = {k: v for k, v in s.missing_inputs().items() if v is not None}
    print("stages awaiting recordings:")
    for stage, rel in sorted(miss.items()):
        print(f"  {stage}: {rel}")
    return 0


def _cmd_qualify(args) -> int:
    s = FprSession(root=args.session)
    q = _qualify.qualify_session(s, save=True)
    print(_qualify.format_report(q))
    return 0 if q.verdict == "PASS" else (2 if q.verdict == "FAIL" else 3)


def _cmd_finger(args) -> int:
    from . import reference as ref
    if args.pi_export:
        pt, pv = ref.read_finger_pi_export(args.csv)
        finger = ref.analyze_finger(pi_t=pt, pi_pct=pv)
    else:
        t, ppg = ref.read_finger_ppg(args.csv)
        finger = ref.analyze_finger(ppg=ppg, t=t)
    print(WATERMARK + "\n")
    print("finger control:", json.dumps(finger.summary(), indent=2,
                                         default=float))
    if args.events:
        events = ref.read_events(args.events)
        win = ref.windows_from_events(events)
        v = ref.stimulus_response(finger, win["baseline"], win["stimulus"])
        print(f"\nstimulus response: {v.verdict} "
              f"(dPI {v.delta_pct:+.1f}% {v.direction}, "
              f"baseline {v.baseline_pi_pct:.2f}% -> stimulus "
              f"{v.stimulus_pi_pct:.2f}%)")
        if v.verdict == "no_response":
            print("  -> the provocation did not reach the periphery: a "
                  "STIMULUS problem, not a facial one")
    return 0


def _cmd_overlay(args) -> int:
    from . import diagnostics
    from .contract import FprSession
    s = FprSession(root=args.session)
    p = diagnostics.render_reference_overlay(s, args.out)
    print(f"wrote reference overlay: {p}")
    return 0


def _cmd_sham_real(args) -> int:
    from . import sham_real, diagnostics
    from .contract import FprSession
    if args.session:
        s = FprSession(root=args.session)
        real = sham_real.trial_from_session(s, "real")
        sham = sham_real.trial_from_session(s, "sham")
        cmp = sham_real.compare_sham_real(real, sham)
        report_dir = s
    elif args.real and args.sham:
        real_s, sham_s = FprSession(root=args.real), FprSession(root=args.sham)
        cmp = sham_real.compare_sessions(real_s, sham_s)
        report_dir = real_s
    else:
        print("provide --session <dir> (named phases) or --real <dir> "
              "--sham <dir> (two sessions)")
        return 2
    print(sham_real.format_comparison(cmp))
    report_dir.ensure_dirs()
    report_dir.save_json("report/sham_real.json", cmp.to_dict())
    if args.out:
        diagnostics.sham_real_overlay(cmp, args.out)
        print(f"\nwrote overlay: {args.out}")
    return 0


def _cmd_diag(args) -> int:
    from . import diagnostics
    made = diagnostics.render_b7(args.out)
    if not made:
        print("no diagnostics rendered (matplotlib unavailable)")
        return 1
    print("wrote:")
    for p in made:
        print(f"  {p}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fpr", description=WATERMARK)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("b7", help="run the synthetic PI pipeline test").set_defaults(
        func=_cmd_b7)

    b0 = sub.add_parser("b0", help="quantisation-headroom calc (pre-purchase)")
    b0.add_argument("--bit-depth", type=int, default=8)
    b0.add_argument("--roi", type=int, default=150, help="ROI side in px")
    b0.add_argument("--expected", type=float, default=0.001,
                    help="smallest expected AC/DC (fraction; 0.001 = 0.1%%)")
    b0.set_defaults(func=_cmd_b0)

    ini = sub.add_parser("init", help="create a session directory skeleton")
    ini.add_argument("session")
    ini.set_defaults(func=_cmd_init)

    q = sub.add_parser("qualify", help="run B0-B8 and write the verdict")
    q.add_argument("session")
    q.set_defaults(func=_cmd_qualify)

    d = sub.add_parser("diagnostics", help="render the B7 diagnostic figures")
    d.add_argument("out")
    d.set_defaults(func=_cmd_diag)

    fg = sub.add_parser("finger", help="inspect a finger-PPG control file")
    fg.add_argument("csv")
    fg.add_argument("--pi-export", action="store_true",
                    help="input is a device PI export (t_s,pi_percent)")
    fg.add_argument("--events", help="events.csv to derive the stimulus window")
    fg.set_defaults(func=_cmd_finger)

    ov = sub.add_parser("overlay", help="render the facial PI + finger + HR + "
                        "temperature reference overlay from a session")
    ov.add_argument("session")
    ov.add_argument("out", help="output PNG path")
    ov.set_defaults(func=_cmd_overlay)

    sr = sub.add_parser("sham-real", help="compare sham vs real trials "
                        "(expectancy/artifact control)")
    sr.add_argument("--session", help="one session with named sham_*/real_* "
                    "event phases")
    sr.add_argument("--real", help="real (cold) session dir (two-session mode)")
    sr.add_argument("--sham", help="sham (thermoneutral) session dir")
    sr.add_argument("--out", help="optional overlay PNG path")
    sr.set_defaults(func=_cmd_sham_real)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main())
