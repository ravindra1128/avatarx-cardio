"""
AvatarX AFib — timing error budget and beat-count floor.

Reconciles two prior AvatarX documents:
  * Greenfield §4.3 quantisation arithmetic: SD(peak)=(1000/f)/sqrt(12),
    SD(IBI)=sqrt(2)*SD(peak), SD(dIBI)=sqrt(6)*SD(peak).
  * AFib Feasibility §5.4 measured result: PhysFormer on a noise-free
    synthetic 30 fps stimulus returned RMSSD 65.9 ms against a true 33.8 ms.

Question: how much of that measured error is irreducible frame quantisation,
and how much is model/peak-detector jitter that engineering can remove?
"""
import numpy as np

def sd_peak(fps):   return (1000.0 / fps) / np.sqrt(12)
def sd_ibi(fps):    return np.sqrt(2) * sd_peak(fps)
def sd_dibi(fps):   return np.sqrt(6) * sd_peak(fps)

# RMSSD observed = sqrt(true^2 + jitter_dIBI^2)  (independent additive timing noise)
def rmssd_obs(true_rmssd, sd_d): return np.sqrt(true_rmssd**2 + sd_d**2)
def inflation(true_rmssd, fps):
    return 100*(rmssd_obs(true_rmssd, sd_dibi(fps))/true_rmssd - 1)

print("="*88)
print(" 1. QUANTISATION FLOOR — irreducible timing noise from frame rate alone")
print("="*88)
print(f"{'fps':>5}{'frame ms':>10}{'SD(peak)':>10}{'SD(IBI)':>10}{'SD(dIBI)':>11}"
      f"{'sinus 30ms':>13}{'sinus 50ms':>13}{'AF 150ms':>11}")
for f in (30, 60, 84, 120, 240):
    print(f"{f:>5}{1000/f:>10.1f}{sd_peak(f):>10.1f}{sd_ibi(f):>10.1f}{sd_dibi(f):>11.1f}"
          f"{inflation(30,f):>12.1f}%{inflation(50,f):>12.1f}%{inflation(150,f):>10.1f}%")

print("""
The asymmetry is the whole problem: quantisation inflates REGULAR rhythms and
leaves IRREGULAR ones alone, so it erodes the AF/sinus margin from the sinus side.
Sub-frame interpolation is the mitigation; frame rate is the ceiling on it.
""")

print("="*88)
print(" 2. DECOMPOSITION of the measured AFib-report result (PhysFormer, 30 fps)")
print("="*88)
TRUE, MEAS, FPS = 33.8, 65.9, 30
q = sd_dibi(FPS)
pred = rmssd_obs(TRUE, q)
excess = np.sqrt(max(MEAS**2 - pred**2, 0))
total_noise = np.sqrt(max(MEAS**2 - TRUE**2, 0))
print(f"  true RMSSD                          {TRUE:6.1f} ms")
print(f"  quantisation SD(dIBI) @30 fps       {q:6.1f} ms")
print(f"  => predicted if quantisation ONLY   {pred:6.1f} ms   (+{100*(pred/TRUE-1):.0f}%)")
print(f"  measured (AFib report §5.4)         {MEAS:6.1f} ms   (+{100*(MEAS/TRUE-1):.0f}%)")
print(f"  total injected timing noise         {total_noise:6.1f} ms")
print(f"  of which quantisation               {q:6.1f} ms  ({100*q**2/total_noise**2:4.1f}% of variance)")
print(f"  of which MODEL / PEAK-DETECTOR      {excess:6.1f} ms  ({100*excess**2/total_noise**2:4.1f}% of variance)")
print(f"""
  Reading: at 30 fps roughly {100*q**2/total_noise**2:.0f}% of the injected variance was the frame
  clock and {100*excess**2/total_noise**2:.0f}% was the extraction stack. Both must be attacked, and the
  larger term is the ENGINEERABLE one. Moving to 60 fps alone would leave
  {rmssd_obs(TRUE, np.sqrt(sd_dibi(60)**2 + excess**2)):.1f} ms; fixing the extractor to a 10 ms residual at 30 fps
  would leave {rmssd_obs(TRUE, np.sqrt(q**2 + 10**2)):.1f} ms. Doing both leaves {rmssd_obs(TRUE, np.sqrt(sd_dibi(60)**2 + 10**2)):.1f} ms.
""")

print("="*88)
print(" 3. WHAT IBI ACCURACY DOES AF/SINUS SEPARATION REQUIRE?")
print("="*88)
print("  Decision boundary sits between sinus RMSSD ~30-50 ms and AF RMSSD >100 ms.")
print("  Requirement: observed sinus must stay below the boundary after inflation.\n")
print(f"  {'sinus':>7}{'boundary':>10}{'max SD(dIBI)':>14}{'-> max SD(peak)':>17}{'min fps if no interp':>22}")
for sinus, bound in [(30, 80), (50, 80), (50, 100), (65, 100)]:
    max_d = np.sqrt(max(bound**2 - sinus**2, 0))
    max_p = max_d / np.sqrt(6)
    min_f = 1000.0 / (max_p * np.sqrt(12))
    print(f"  {sinus:>7}{bound:>10}{max_d:>14.1f}{max_p:>17.1f}{min_f:>22.0f}")
print("""
  Even the tightest row is satisfied by ~30 fps on paper. The binding constraint is
  therefore NOT frame rate in isolation — it is total timing error including the
  extractor. That is why the beat-level gate below is expressed as measured IBI MAE
  against ECG, not as a frame rate.
""")

print("="*88)
print(" 4. BEAT-COUNT FLOOR — why 30 s is below the estimators' own requirements")
print("="*88)
print(f"  {'scan':>6}{'beats@50':>10}{'@60':>7}{'@75':>7}{'@100':>7}{'@120':>7}   {'estimator status at 60 bpm':<40}")
for s in (15, 30, 45, 60, 90, 120, 300):
    b = {hr: int(s*hr/60) for hr in (50, 60, 75, 100, 120)}
    n = b[60]
    if   n < 25:  st = "below every published threshold"
    elif n < 40:  st = "above 25-beat knee, below 85-beat optimum"
    elif n < 85:  st = "approaching optimum"
    elif n < 120: st = "at/past 85-beat optimum"
    else:         st = "saturated; UX cost only"
    print(f"  {s:>4}s{b[50]:>10}{b[60]:>7}{b[75]:>7}{b[100]:>7}{b[120]:>7}   {st:<40}")
print("""
  Sample entropy with m=2 conventionally wants N >= 100-200 points for a stable
  estimate; Poincare SD1/SD2 are second-moment statistics and are usable earlier;
  histogram/density-based irregularity indices are the most data-hungry.
  A 30 s scan at 60 bpm yields ~30 intervals -- it under-serves precisely the
  entropy-family features that carry most of the AF discrimination.
""")

print("="*88)
print(" 5. SERIAL CONFIRMATION — the Apple 5-of-6 pattern applied to a face scan")
print("="*88)
from math import comb
def kofn(p, k, n): return sum(comb(n, i)*p**i*(1-p)**(n-i) for i in range(k, n+1))
def ppv(se, sp, prev): return prev*se/(prev*se + (1-prev)*(1-sp))
print(f"  Base per-scan operating point: Se 0.850, Sp 0.940 (planning assumption)\n")
print(f"  {'rule':>10}{'Se_eff':>9}{'Sp_eff':>9}{'PPV@1%':>9}{'PPV@2%':>9}{'PPV@5%':>9}{'FP/1000@1%':>12}")
SE, SP = 0.85, 0.94
for k, n in [(1,1),(2,2),(2,3),(3,3),(3,4),(4,5),(5,6)]:
    se_e, sp_e = kofn(SE,k,n), 1-kofn(1-SP,k,n)
    fp = 1000*(1-0.01)*(1-sp_e)
    print(f"  {f'{k} of {n}':>10}{se_e:>9.3f}{sp_e:>9.4f}{ppv(se_e,sp_e,0.01):>9.3f}"
          f"{ppv(se_e,sp_e,0.02):>9.3f}{ppv(se_e,sp_e,0.05):>9.3f}{fp:>12.1f}")
print("""
  2-of-3 is the sweet spot: it costs ~6 points of sensitivity and buys ~an order of
  magnitude in PPV. Apple's cleared 5-of-6 is stricter because a watch gets unlimited
  attempts; a face scan should not demand six sittings.
""")
