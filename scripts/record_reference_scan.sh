#!/bin/bash
# Lossless reference recording from the Mac camera — the control arm of the
# phone-vs-laptop comparison.
#
# WHY: webapp phone clips reach cross-ROI correlation ~0.16; lab-rig clips
# reach ~0.33. Cross-ROI agreement is what lets beat fusion match beats across
# regions, and it is the measured bottleneck behind coverage 0.47 and a 2%
# ACCEPT rate. But the two sets differ in more than the camera — different
# sessions, lighting, possibly subject. This records the SAME person in the
# SAME light so the phone can be compared against it fairly.
#
# Encodes FFV1 (mathematically lossless) at 480x360, matching the existing
# demo_*.avi corpus clips, so the result is directly comparable to them.
#
# USAGE:  scripts/record_reference_scan.sh [seconds] [label]
set -euo pipefail

SECS="${1:-60}"
LABEL="${2:-ref}"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/controlled"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_DIR/laptop_${LABEL}_${STAMP}.avi"
mkdir -p "$OUT_DIR"

cat <<TXT
Recording ${SECS}s of LOSSLESS video to:
  $OUT

Hold the SAME position and lighting you will use for the phone scan that
follows. Sit still, face the camera, normal breathing. The point is to change
ONLY the camera between the two recordings — so do not move the lamp, open a
blind, or change seats in between.

Starting in 3 seconds...
TXT
sleep 3

ffmpeg -hide_banner -loglevel warning \
  -f avfoundation -framerate 30 -video_size 1280x720 -i "0" \
  -t "$SECS" \
  -vf "scale=480:360" \
  -c:v ffv1 -level 3 -pix_fmt yuv420p -an \
  "$OUT"

SIZE=$(du -h "$OUT" | cut -f1)
cat <<TXT

Done: $OUT  ($SIZE)

NEXT — the paired phone scan, within the next few minutes, same seat, same light:
  1. open  https://staging.theavatarx.ai/beta/cardio-staging  on the phone
  2. run one scan to completion
  3. note the time; the staging sheet tab 'cardio-staging' will hold its row

Then compare:
  python .claude/skills/cardio-pipeline-optimizer/scripts/../../../../scripts/compare_capture.py \\
      "$OUT"
TXT
