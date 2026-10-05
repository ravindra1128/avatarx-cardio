"""Read-only diagnostic replay; copies videos before the destructive prep path.

Run from repo root with miniconda Python. No sheet writes or production edits.
Output contains intermediate amplitudes so hypotheses can be tested without
decoding a second time. Timestamp proximity is NOT verified subject identity.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["AFIB_SHEET_ENABLED"] = "0"
os.environ["AFIB_TRACE_PATH"] = "0"
os.environ["AFIB_MAX_COLLAPSED_FRACTION"] = "0.05"
from app.measure_api import measure_video_details
from features import hemodynamics as h


def spread(v):
    a = np.asarray(v, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0}
    m = float(a.mean())
    return dict(n=int(a.size), mean=m, sd=float(a.std(ddof=1)) if a.size > 1 else None,
                min=float(a.min()), max=float(a.max()), range=float(np.ptp(a)),
                cv=float(a.std(ddof=1)/abs(m)) if a.size > 1 and m else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', nargs='?', default=str(ROOT/'research/vascular_tone_audit/replay.json'))
    parser.add_argument('--phone-only', action='store_true')
    args = parser.parse_args()
    target = Path(args.out)
    records = json.loads((ROOT/'data/eval_cache/eval_manifest.json').read_text())['records']
    if args.phone_only:
        records = [r for r in records if r['id'].startswith(('20260916', '20260917'))]
    rows = []
    original_segment, original_pool = h.segment_amplitudes, h.pool_amplitudes
    for record in records:
        segments, pools = [], []

        def segment(waves, ts, windows):
            out = original_segment(waves, ts, windows)
            raw = {roi: [[float(ts[a]), float(np.ptp(wave[a:b]))]
                         for window in windows.get(roi, []) for a,b in [window[:2]]
                         if b-a >= 4 and np.all(np.isfinite(wave[a:b]))]
                   for roi, wave in waves.items()}
            segments.append(dict(start=float(ts[0]), end=float(ts[-1]), raw=raw, normalized=out))
            return out

        def pool(per_roi):
            out = original_pool(per_roi)
            counts = [sum(any(abs(tt-t)<0.05 for tt, _ in series)
                          for series in per_roi.values() if len(series)>=5) for t, _ in out]
            pools.append(dict(series=out, contributors=counts, per_roi=per_roi))
            return out

        h.segment_amplitudes, h.pool_amplitudes = segment, pool
        source = ROOT / record['path']
        before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        row = dict(id=record['id'], split=record['split'], sha256=before_hash,
                   subject='unverified', device='unverified', conditions='unverified',
                   timestamp=record['id'][:16] if record['id'].startswith('2026') else None)
        try:
            with tempfile.TemporaryDirectory(prefix='tone-audit-') as d:
                copied = Path(d)/source.name
                shutil.copy2(source, copied)
                side = Path(str(source)+'.timestamps.json')
                if side.exists():
                    shutil.copy2(side, str(copied)+'.timestamps.json')
                doc, det = measure_video_details(str(copied), manifest={'capture_profile':'consumer'},
                                                window_s=0 if record.get('pretrimmed') else 70,
                                                scale='480x360')
            item = next((x for x in doc.get('biomarkers', {}).get('items', [])
                         if x['key']=='vascular_tone'), {})
            ing = det.get('ingest')
            raw_stats = {}
            if ing is not None:
                for roi, rgb in ing.traces.items():
                    rgb = np.asarray(rgb)
                    raw_stats[roi] = {c:spread(rgb[:,i]) for i,c in enumerate('RGB')}
            row.update(returned=True, outcome=doc.get('outcome'), tone=item,
                       evidence=det.get('evidence'), sqi=doc.get('signal_quality_index'),
                       raw_rgb=raw_stats, segments=segments, pools=pools,
                       trim=doc.get('trim'), clock=doc.get('clock'), timing=doc.get('timing'))
        except Exception as e:
            row.update(returned=False, error=f'{type(e).__name__}: {e}')
        finally:
            h.segment_amplitudes, h.pool_amplitudes = original_segment, original_pool
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash, 'corpus mutated'
        rows.append(row)
        target.write_text(json.dumps(dict(window_s=70, scale='480x360',
                                        scope='local checkout; retained clips, not live UI', rows=rows),
                                    indent=2, default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
        print('[tone-audit]',row['id'],row.get('tone',{}).get('value'),row.get('error',''),flush=True)


if __name__ == '__main__':
    main()
