#!/usr/bin/env python3
"""Console workflow of 2026-09-01: load a DO out_grid.h5 and compare the
per-realisation asymmetry measures for one particle P:
- M1  = <u_p, phi_1> = a(t) + Y1     (mode-1 projection)
- Mym = sign(v_p(x=2)) sqrt(int v_p^2 dx) on the centerline (Ducimetiere Eq. 25)
- dL  = L_up - L_low                 (recirculation bubble length difference,
                                     bubble = near-wall u < 0 on the second /
                                     penultimate rows)
- dLs = the same with the bubble = wall shear du/dy < 0 (towards the fluid)
                                     on the same rows, central difference
- Yb  = barycentre of the velocity deficit U_REF - u (U_REF = 1, inlet
                                     centreline) over x > 0, particle P only
- Y1 alone
Display scale factors are in the labels; tweak them to taste.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

PATH = '/home/isma/nektar_src_full/cases/se/runs/2026_08_16_do_phi22_t3000_jed/output/out_grid.h5'
OUT = '/home/isma/todisimo/studies/7.0_epfl/year2/academic/my_notes/1_sep/18_sep_1.pdf'
P = 6

# -- loading as np arrays (the full mode arrays are 2 x 15.7 GB, so only the
#    rows each measure needs are read: mode 1 for M1, the centerline for Mym,
#    the near-wall rows for dL; u_i / v_i below are h5py datasets, sliced on use)
f = h5py.File(PATH, 'r')
u = np.array(f['u'])
v = np.array(f['v'])
ys = np.array(f['yi'])
u_i = f['mode_u']
v_i = f['mode_v']
t = np.array(f['t'])
x = np.array(f['x'])
y = np.array(f['y'])


# -- total field of one realisation (full grid, ~2x2.6 GB; call on demand)
def total_field(p):
    up = u + np.einsum('jyxt,jt->yxt', np.array(u_i), ys[p])
    vp = v + np.einsum('jyxt,jt->yxt', np.array(v_i), ys[p])
    return up, vp


# -- M1: projection on mode 1
dA = (x[0, 1] - x[0, 0]) * (y[1, 0] - y[0, 0])
a = (np.einsum('yxt,yxt->t', u, u_i[0]) + np.einsum('yxt,yxt->t', v, v_i[0])) * dA
M1 = a + ys[:, 0, :]

# -- Mym: Eq. 25 per realisation (quadratic in Y on the centerline)
iy = np.abs(y[:, 0]).argmin()
xm = x[0] >= 0
xg = x[0, xm]
vb = v[iy, xm, :]
vm = v_i[:, iy][:, xm, :]
A = trap(vb**2, xg, axis=0)
b = trap(vb[None] * vm, xg, axis=1)
C = trap(vm[:, None] * vm[None], xg, axis=2)
M2 = A + 2 * np.einsum('pjt,jt->pt', ys, b) + np.einsum('pjt,jkt,pkt->pt', ys, C, ys)
i2 = np.abs(xg - 2.0).argmin()
Mym = np.sign(vb[i2] + np.einsum('pjt,jt->pt', ys, vm[:, i2, :])) * np.sqrt(M2)

# -- Yb: barycentre of the velocity deficit over the expansion (x > 0), the
#    transposition of the wake-deficit barycentre of the Ahmed-body
#    bi-stability (Grandemange et al. 2013: z_b = iint z (Uinf - U) dS /
#    iint (Uinf - U) dS over the wake U < Uinf). Reference = inlet centreline
#    velocity U_REF = 1 (the largest velocity of the flow):
#        Yb = iint_{u<U_REF} y (U_REF - u) dx dy / iint_{u<U_REF} (U_REF - u) dx dy.
#    Over the y-symmetric domain iint y U_REF dA = 0 and iint u dA = Q X is
#    constant (modes carry no flux), so U_REF and the x-extent only set the
#    scale; the signal is -iint y u dA (linear in Y) as long as u < U_REF
#    everywhere. Computed exactly (with the mask) for particle P.
U_REF = 1.0
cb = np.where(x[0] > 0)[0]
c0, c1 = cb[0], cb[-1] + 1
yy = y[:, 0]
uP = u[:, c0:c1, :].copy()                                  # (y, x, t) of particle P, one mode at a time
for j in range(ys.shape[1]):
    uP += u_i[j, :, c0:c1, :] * ys[P, j, :]
u_exceeds = int((uP >= U_REF).sum())
wb = np.where(uP < U_REF, U_REF - uP, 0.0)
Yb = np.einsum('y,yxt->t', yy, wb) / np.einsum('yxt->t', wb)
del uP, wb
print(f'Yb: grid points with u >= U_REF over all t: {u_exceeds} (mask inactive if 0)')

# -- dL: bubble length difference (near-wall u sign change, chunked in time)
xm2 = x[0] > 0
xg2 = x[0, xm2]


def L_wall(iy_w):
    Aw = u[iy_w, xm2, :]
    Bw = u_i[:, iy_w][:, xm2, :]
    L = np.empty((ys.shape[0], t.size))
    for k in range(0, t.size, 100):
        s = slice(k, min(k + 100, t.size))
        uw = Aw[None, :, s] + np.einsum('pjt,jxt->pxt', ys[:, :, s], Bw[:, :, s])
        # reattachment = downstream END of the backflow (last u<0 column);
        # the first column x=0.05 is always slightly positive, so a
        # first-positive search saturates there
        L[:, s] = xg2[uw.shape[1] - 1 - np.argmax((uw < 0)[:, ::-1, :], axis=1)]
    return L


dL = L_wall(y.shape[0] - 2) - L_wall(1)


# -- dLs: same rows, bubble = wall shear du/dy < 0 with y pointing into the
#    fluid (sgn = +1 lower wall, -1 upper wall); du/dy by central difference
#    between the neighbouring rows (the wall row itself has u = 0)
dy = y[1, 0] - y[0, 0]


def L_wall_shear(iy_w, sgn):
    Aw = sgn * (u[iy_w + 1, xm2, :] - u[iy_w - 1, xm2, :]) / (2 * dy)
    Bw = sgn * (u_i[:, iy_w + 1][:, xm2, :] - u_i[:, iy_w - 1][:, xm2, :]) / (2 * dy)
    L = np.empty((ys.shape[0], t.size))
    for k in range(0, t.size, 100):
        s = slice(k, min(k + 100, t.size))
        tw = Aw[None, :, s] + np.einsum('pjt,jxt->pxt', ys[:, :, s], Bw[:, :, s])
        L[:, s] = xg2[tw.shape[1] - 1 - np.argmax((tw < 0)[:, ::-1, :], axis=1)]
    return L


dLs = L_wall_shear(y.shape[0] - 2, -1) - L_wall_shear(1, +1)

# -- comparison plot for particle P
fig, ax = plt.subplots(figsize=(15, 7.5))
ax.plot(t, M1[P, :].T, label=rf"$M_{{{P}}}$ (mode 1)")
ax.plot(t, 10 * Mym[P, :].T, label=rf"$10M^{{\rm YM}}_{{{P}}}$ (Eq. 25)")
ax.plot(t, 0.5 * dL[P, :].T, label=rf"$0.5\,\Delta L_{{{P}}}$ (last $u < 0$)")
ax.plot(t, 0.5 * dLs[P, :].T, label=rf"$0.5\,\Delta L_{{{P}}}$ (last $\partial u/\partial y < 0$)")
ax.plot(t, ys[P, 0, :].T, label=rf"$Y_1^{{({P})}}$")
ax.plot(t, 50 * Yb, label=rf"$50\,y_b^{{({P})}}$ (deficit barycentre)")
ax.axhline(0, color='grey', ls='--')
ax.set_xlabel("t")
ax.set_ylabel("value")
ax.legend(fontsize=15)
fig.savefig(OUT, bbox_inches='tight')
plt.show()
