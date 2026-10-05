"""Statistics and deterministic feature perturbations; no clinical tolerance assumed."""
import json
from pathlib import Path
from datetime import datetime
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from features.hemodynamics import pool_amplitudes, vasomotor_indices


def stats(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    if not len(a): return {'n':0}
    sd = float(a.std(ddof=1)) if len(a)>1 else None
    return dict(n=len(a),mean=float(a.mean()),sd=sd,min=float(a.min()),max=float(a.max()),
                range=float(np.ptp(a)),cv=sd/abs(float(a.mean())) if sd is not None and a.mean() else None)


def score(a):
    return vasomotor_indices(a, False).get('score')


def main():
    base = ROOT/'research/vascular_tone_audit'
    source=Path(sys.argv[1]) if len(sys.argv)>1 else base/'replay.json'
    target=Path(sys.argv[2]) if len(sys.argv)>2 else base/'analysis.json'
    doc=json.loads(source.read_text())
    rows=doc['rows']; scan=[]
    for row in rows:
        p=(row.get('pools') or [{}])[0]; series=p.get('series',[])
        r=dict(id=row['id'],split=row['split'],value=row.get('tone',{}).get('value'),
               tier=row.get('tone',{}).get('tier'),n_amplitudes=len(series),
               contributors=stats(p.get('contributors',[])),
               single_roi_fraction=sum(x==1 for x in p.get('contributors',[]))/len(series) if series else None,
               sqi=row.get('sqi'),coherence=(row.get('evidence') or {}).get('cross_roi_coherence'),
               per_roi_cv={roi:stats([a for _,a in v]).get('cv') for roi,v in p.get('per_roi',{}).items()},
               rgb_green_mean={roi:channels['G'].get('mean') for roi,channels in row.get('raw_rgb',{}).items()})
        if series:
            t=np.asarray([t for t,a in series]); a=np.asarray([a for t,a in series])
            phase=(t-t[0])/max(t[-1]-t[0],1e-9)
            r['perturbations']={
                'baseline':score(series),
                'constant_amplitude_gain_10pct':score(list(zip(t,1.1*a))),
                'amplitude_gain_ramp_0_to_10pct':score(list(zip(t,a*(1+.1*phase)))),
                'one_amplitude_plus_10pct_max_abs_points':max(abs(score(list(zip(t,a*(1+.1*(np.arange(len(a))==i)))))-score(series)) for i in range(len(a))),
            }
            rois=p.get('per_roi',{})
            r['perturbations']['roi_foot_jitter_20ms_max_abs_points']=max(
                abs(score(pool_amplitudes({k:[(tt+(.02 if k==roi else 0),aa) for tt,aa in v]
                                           for k,v in rois.items()}))-score(series))
                for roi in rois)
        scan.append(r)
    confirmed=[r for r in scan if r['id'].startswith(('20260916','20260917'))]
    pairs=[]
    for a,b in zip(confirmed,confirmed[1:]):
        gap=(datetime.strptime(b['id'][:16],'%Y%m%dT%H%M%SZ')-datetime.strptime(a['id'][:16],'%Y%m%dT%H%M%SZ')).total_seconds()/60
        if 0<gap<=20:
            x,y=a['value'],b['value'];good=x is not None and y is not None
            pairs.append(dict(a=a['id'],b=b['id'],gap_min=gap,values=[x,y],both_computed=good,
                              abs_difference=abs(x-y) if good else None,
                              relative_difference=abs(x-y)/((x+y)/2) if good and x+y else None))
    out=dict(scope=doc['scope'],identity='Owner confirmed September 16–17 same person/phone/comparable rest in chat',
             coverage=dict(all_n=len(rows),all_computed=sum(r['value'] is not None for r in scan),
                           confirmed_n=len(confirmed),confirmed_computed=sum(r['value'] is not None for r in confirmed)),
             confirmed_value_spread=stats([r['value'] for r in confirmed]),
             pairs=pairs,pair_absolute_difference=stats([p['abs_difference'] for p in pairs]),
             exploratory_20pct_pass_n=sum(p['relative_difference'] is not None and p['relative_difference']<=.2 for p in pairs),
             clinical_tolerance=None,scans=scan)
    target.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ('scans','pairs')},indent=2))


if __name__=='__main__': main()
