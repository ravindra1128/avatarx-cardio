"""Telemetry describes existing decisions without altering the measured signal."""
import json
import numpy as np
import pytest
from beats.detector import Beat, BeatSeries
from beats.ibi import clean_runs
from inference import evidence
from app import scan_evidence, result_sheet


def series(rr, confidence=None, segments=None):
    ts = np.r_[0, np.cumsum(rr) / 1000]
    return BeatSeries([Beat(float(t), confidence[i] if confidence else 1., 1., 1., 1., 1.,
                           segment=segments[i] if segments else 0) for i,t in enumerate(ts)],30,float(ts[-1]))


def partition(rs):
    a=rs.rejection_audit
    assert a['adjacent_intervals'] == sum(a[k] for k in (
        'confidence_intervals','capture_gap_intervals','range_intervals',
        'splitter_intervals','short_run_intervals','kept_intervals'))
    assert a['kept_intervals'] == rs.n_intervals
    json.dumps(a,allow_nan=False)
    return a


def test_clean_scan_has_no_rejections():
    a=partition(clean_runs(series([800]*20)))
    assert a['adjacent_intervals']==a['kept_intervals']==20


def test_confidence_gap_and_range_counts_do_not_overlap():
    s=series([800,800,800,3000,800,800,800,800],
             confidence=[1,0,1,1,1,1,1,1,1],segments=[0,0,0,1,1,1,1,1,1])
    a=partition(clean_runs(s))
    assert a['confidence_intervals']==2
    assert a['capture_gap_intervals']==1
    assert a['range_intervals']==1


def test_splitter_exclusions_are_separate_from_short_run_loss():
    a=partition(clean_runs(series([800]*5+[400,400]+[800]*5)))
    assert a['splitter_intervals']==2
    b=partition(clean_runs(series([800]*5),min_run_beats=7))
    assert b['short_run_intervals']==5 and b['kept_intervals']==0


@pytest.mark.parametrize('n',[0,1])
def test_tiny_input_is_accounted(n):
    s=series([]); s.beats=s.beats[:n]
    assert partition(clean_runs(s))['adjacent_intervals']==0


def test_counter_partition_for_varied_scans():
    for seed in range(100):
        rng=np.random.default_rng(seed)
        rr=rng.uniform(150,2500,100).tolist()
        s=series(rr,confidence=rng.uniform(0,1,101).tolist(),segments=[i//15 for i in range(101)])
        partition(clean_runs(s,min_conf=.3,min_run_beats=3))


def test_half_rate_audit_reports_both_peaks():
    f=np.arange(81)*.05
    p=.01+.7*np.exp(-.5*((f-1)/.02)**2)+np.exp(-.5*((f-2)/.02)**2)
    before=evidence._fundamental(f,p)
    out=evidence._peak_diagnostics(f,p)
    assert out['strongest_peak_bpm']==120
    assert out['selected_peak']['bpm']==60
    assert out['selection']=='subharmonic'
    assert out['selected_peak']['relative_to_strongest']==pytest.approx(.71/1.01)
    assert before==evidence._fundamental(f,p)
    json.dumps(out,allow_nan=False)


def test_drift_boundary_and_no_peak_are_explicit():
    f=np.arange(81)*.05;p=1/(f+.1)**2
    out=evidence._peak_diagnostics(f,p)
    assert out['selection']=='no_peak' and out['selected_peak'] is None
    assert out['top_peaks']==[]
    assert out['band_max_bpm']==42


def test_peak_summary_is_bounded_and_preserves_source_arrays():
    f=np.linspace(0,4,1001);p=1+np.sin(f*200)
    before=p.copy();out=evidence._peak_diagnostics(f,p)
    assert len(out['top_peaks'])<=5
    np.testing.assert_array_equal(before,p)
    assert out['peak_count']>5


def test_diagnostic_failure_does_not_break_spectral_selection(monkeypatch):
    # The diagnostic itself catches its reporting failure; production selection remains valid.
    original=evidence._fundamental
    count=[0]
    def intermittent(f,p):
        count[0]+=1
        if count[0]%2==0:raise ValueError('diagnostic only')
        return original(f,p)
    monkeypatch.setattr(evidence,'_fundamental',intermittent)
    t=np.arange(1500)/30
    out=evidence.spectral_pulse({r:np.sin(2*np.pi*1.5*t) for r in evidence.ROI_NAMES},30)
    assert out['pulse_spectral_bpm']==90
    assert out['spectral_diagnostics']['fused']['state']=='assessment_failed'


def test_service_and_sheet_keep_scalar_audits():
    rs=clean_runs(series([800]*20))
    vd=scan_evidence.video_duration_summary({}, {'runset':rs})
    assert vd['interval_rejections']['kept_intervals']==20
    t=np.arange(1500)/30
    ev=evidence.spectral_pulse({r:np.sin(2*np.pi*1.5*t) for r in evidence.ROI_NAMES},30)
    row=result_sheet.row_from_doc({'signal_quality_index':.29996,'debug':{'evidence':ev,'video_duration':vd}})
    assert row['Spectral Strongest bpm']==90
    assert json.loads(row['Video Interval Rejections'])['kept_intervals']==20
    assert row['Video SQI Exact']==.29996
    assert len(row['Spectral Peak Audit'])<6000
    assert len(result_sheet.COLUMNS)==len(set(result_sheet.COLUMNS))
    assert result_sheet.row_from_doc({})['Spectral Peak Audit']==''
