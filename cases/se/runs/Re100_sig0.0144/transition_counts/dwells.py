"""Which crossings of A does the DO count at 0.5*A_fp that the count at A_fp does not?

For every particle: Schmitt trigger at h = 0.5*A_fp gives crossing times and
dwells (segments in one state between two consecutive crossings). For every
dwell: duration, deepest value s*A reached, whether it reaches A_fp.
A dwell that never reaches A_fp is invisible to the trigger at A_fp, so it
removes two crossings there (in and back out).
"""
import glob, sys
import h5py, numpy as np

W = 0.701
TMIN = 300.0
H = 0.5 * W


def dwells(t, a):
    """Segments in state +-1 between consecutive crossings at H (interior dwells only)."""
    s = np.where(a > H, 1, np.where(a < -H, -1, 0))
    st = 0; out = []; k0 = None
    for k in range(len(t)):
        if s[k] != 0 and s[k] != st:
            if st != 0:
                seg = a[k0:k] * st          # signed toward the occupied well
                out.append((t[k] - t[k0], seg.max(), seg.min()))
            st = s[k]; k0 = k
    return out


def report(name, t, A, dt):
    w = t >= TMIN
    t, A = t[w], A[w]
    D = []
    for p in range(A.shape[1]):
        D += dwells(t, A[:, p])
    D = np.array(D)                          # (n, 3): duration, max s*A, min s*A
    dur, mx = D[:, 0], D[:, 1]
    reach = mx >= W
    nr = ~reach
    print(f"\n{name}: {len(D)} interior dwells (sampling dt={dt})")
    print(f"  reach A_fp: {reach.sum()}   do not: {nr.sum()} ({100*nr.mean():.1f}%)")
    if nr.sum():
        q = np.percentile(dur[nr], [5, 50, 95])
        print(f"  non-reaching dwells: duration p5/p50/p95 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} t.u., max {dur[nr].max():.0f}")
        q = np.percentile(mx[nr] / W, [5, 50, 95])
        print(f"                       deepest s*A/A_fp p5/p50/p95 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")
        print(f"                       duration <= 2dt: {(dur[nr] <= 2*dt).sum()},  <= 20 t.u.: {(dur[nr] <= 20).sum()},  > 100 t.u.: {(dur[nr] > 100).sum()}")
    q = np.percentile(dur[reach], [5, 50, 95])
    print(f"  reaching dwells: duration p5/p50/p95 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} t.u.")
    # in-well statistics: samples with |A| > H, per state
    aw = np.abs(A[np.abs(A) > H])
    print(f"  in-well |A| (|A|>0.5 A_fp): median {np.median(aw):.3f}  std {aw.std():.3f}  p5 {np.percentile(aw,5):.3f}  p95 {np.percentile(aw,95):.3f}")


root = '/home/isma/nektar_src_full/cases/se/runs/2026_09_13_jed_sweep/'
for f in sorted(glob.glob(root + 'dns_Re100_sig0.0144_seed*/output/extract/grid.h5')):
    g = h5py.File(f); t = g['t'][:]; A = g['A'][:][:, None]
    report(f.split('/')[-4], t, A, t[1] - t[0])
for d in ['do_Re100_sig0.0144_S6_Np1000_seed45', 'do_Re100_sig0.0144_S6_Np1000_seed45_eps0.0001']:
    g = h5py.File(root + d + '/output/extract/measures.h5'); t = g['t'][:]
    A = g['A0'][:][:, None] + np.einsum('tps,ts->tp', g['yi'][:], g['Ai'][:])
    report(d, t, A, t[1] - t[0])
    # same DO series subsampled to the DNS cadence (every 4 t.u.) to separate the sampling effect
    report(d + '  [subsampled every 4 t.u.]', t[::4], A[::4], 4.0)
