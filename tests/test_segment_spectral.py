import numpy as np
import pytest
from scipy.signal import welch
from inference import evidence as ev


def wave(n=600, bpm=75, offset=0):
    return np.sin(2*np.pi*bpm/60*np.arange(n)/30)+offset


@pytest.mark.parametrize("fps", [24, 30, 30.3, 60])
def test_single_segment_is_bit_identical(fps):
    x=wave(1301)
    f,p=ev._welch_psd(x,fps)
    fs,ps,info=ev._segmented_psd(x,fps,[len(x)])
    np.testing.assert_array_equal(f,fs)
    np.testing.assert_array_equal(p,ps)
    assert info['state']=='assessed'


def test_independent_offsets_cannot_create_a_spectral_peak():
    a,b=wave(360),wave(360)
    f,p,_=ev._segmented_psd(np.r_[a,b],30,[360,360])
    ff,pp,_=ev._segmented_psd(np.r_[a+100,b-100],30,[360,360])
    np.testing.assert_array_equal(f,ff)
    np.testing.assert_allclose(p,pp,rtol=1e-10,atol=1e-12)
    assert ev._fundamental(ff,pp)[0]*60==pytest.approx(75)
    _,joined=ev._welch_psd(np.r_[a+100,b-100],30)
    # Joined windows have enormous nonphysiological boundary energy.
    assert np.max(joined)>np.max(pp)*100


def test_short_fragments_never_add_up_to_a_valid_continuous_window():
    f,p,a=ev._segmented_psd(wave(720),30,[240,240,240])
    assert f is None and p is None
    assert a['eligible_parts']==0 and a['excluded_samples']==720


def test_nonfinite_holes_are_boundaries_not_deleted_time():
    x=np.r_[wave(450),np.nan,wave(450)]
    f,p,a=ev._segmented_psd(x,30,[901])
    assert a['eligible_parts']==2 and a['excluded_samples']==1
    assert ev._fundamental(f,p)[0]*60==pytest.approx(76,abs=2)


@pytest.mark.parametrize('lengths',[[299],[200,200],[-1,301],[150.,150.]])
def test_invalid_layout_never_falls_back_to_joined_data(lengths):
    f,p,a=ev._segmented_psd(wave(300),30,lengths)
    assert f is None and p is None and a['state']=='invalid_segment_layout'


def test_pooling_weights_equal_number_of_non_crossing_welch_windows():
    a,b=wave(600,75),wave(1200,90)
    f,p,info=ev._segmented_psd(np.r_[a,b],30,[600,1200])
    fa,pa=welch(a-a.mean(),fs=30,nperseg=600)
    fb,pb=welch(b-b.mean(),fs=30,nperseg=600)
    np.testing.assert_array_equal(f,fa)
    np.testing.assert_allclose(p,(pa+3*pb)/4)
    assert info['welch_windows']==4 and info['nperseg']==600


def test_extraction_metadata_reaches_spectral_call_without_changing_mapping(monkeypatch):
    ts=np.r_[np.arange(360)/30, 20+np.arange(360)/30]
    monkeypatch.setitem(ev._EXTRACTORS,'test',lambda tr,fps,band=None:np.asarray(tr[:,0]))
    monkeypatch.setattr(ev,'orient_rois_consistently',lambda x:x)
    monkeypatch.setattr(ev,'detect_beats_single_roi',lambda *args:[])
    traces={r:np.tile(np.r_[wave(360),wave(360)][:,None],(1,3)) for r in ev.ROI_NAMES}
    raw,_,segments=ev.extract_and_detect(traces,ts,30,{'decision':{'extractor':'test'},'sqi':{'band_hz':[.7,3]}})
    assert isinstance(raw,dict) and raw.segment_lengths==(360,360)
    assert all(len(v)==720 for v in raw.values())
    out=ev.spectral_pulse(raw,30)
    assert out['pulse_spectral_bpm']==75
    assert out['spectral_diagnostics']['clock']=='within_capture_segments_nominal_fps'
    assert out['spectral_diagnostics']['regions']['forehead']['segments']['eligible_parts']==2


def test_all_short_segments_return_explicit_unavailable_spectrum():
    raw=ev.SegmentedWaveforms({r:wave(720) for r in ev.ROI_NAMES},[240,240,240])
    out=ev.spectral_pulse(raw,30)
    assert out['pulse_spectral_bpm'] is None and out['pulse_spectral_roi_agree']==0
    assert out['spectral_diagnostics']['regions']['nose']['state']=='insufficient_continuous_samples'
