#!/usr/bin/env python3
"""compare_measures.py -- the DOMeasures filter of a run against the python
extraction of the same run.

    compare_measures.py FILTER.h5 PYTHON.h5

FILTER.h5  output/measures/measures.h5 written by FilterDOMeasures (exact
           quadrature on the expansion, format DO_MEASURES_V1).
PYTHON.h5  output/extract/measures.h5 of `extract.sbatch do RUN PHI 1`
           (FieldConvert on the 901 x 121 plane, extract_lib.py reductions,
           assemble.py). KEEP_EVERY = 1 keeps the plane of the mean and the
           modes at every archive: y_b and the wall shear are computed from it.

Rows are matched by step. For every quantity the table gives the largest
|filter - python| over the matched instants (and particles, and points) and
the largest |filter| as the scale. Quantities defined the same way on both
sides (Y_1, A, M, M1, a, G, c, b, C, v at the sensor, the lines) differ by
the plane's quadrature and interpolation error only. Delta L and the wall
shear differ by construction: the filter takes the exact du/dn on the wall
and the end of the first bubble after the corner eddy; the python routes
take the last column with u < 0 on |y| = 1.475 (dL) or with the central
difference du/dn < 0 (dLs).
Only numpy and h5py are required.
"""
import sys

import h5py
import numpy as np

CHUNK = 200                        # particles per block for the per-particle lines


def mx(a):
    a = np.asarray(a, dtype=float)
    return float(np.max(np.abs(a))) if a.size else 0.0


def last_negative(rows, xw):
    """extract_lib.reattachment on rows (..., nx): x of the last column with a
    negative value; none negative -> xw[-1]."""
    neg = rows < 0
    k = rows.shape[-1] - 1 - np.argmax(neg[..., ::-1], axis=-1)
    return xw[k]


def wall_dudn(u, dy, jlow, jup):
    """se.wall_dudn: central difference on the first row off each wall, n into
    the fluid; u (..., NY, NX) -> (..., 2, NX), [0] lower, [1] upper."""
    low = (u[..., jlow + 1, :] - u[..., jlow - 1, :]) / (2 * dy)
    up = -(u[..., jup + 1, :] - u[..., jup - 1, :]) / (2 * dy)
    return np.stack([low, up], axis=-2)


def trap_weights(xp, yrow):
    """Tensor trapezoid weights on the two rectangles of the plane,
    [-5, 0] x [-0.5, 0.5] and [0, 40] x [-1.5, 1.5]; a point on their common
    edge gets the weights of both. Points are located by their integer index."""
    dx, dy = xp[1] - xp[0], yrow[1] - yrow[0]
    W = np.zeros((yrow.size, xp.size))
    for xa_, xb_, ya_, yb_ in ((-5.0, 0.0, -0.5, 0.5), (0.0, 40.0, -1.5, 1.5)):
        i0, i1 = int(round((xa_ - xp[0]) / dx)), int(round((xb_ - xp[0]) / dx))
        j0, j1 = int(round((ya_ - yrow[0]) / dy)), int(round((yb_ - yrow[0]) / dy))
        wx = np.zeros(xp.size)
        wx[i0:i1 + 1] = dx
        wx[[i0, i1]] = 0.5 * dx
        wy = np.zeros(yrow.size)
        wy[j0:j1 + 1] = dy
        wy[[j0, j1]] = 0.5 * dy
        W += wy[:, None] * wx[None, :]
    return W


def interp_rows(xnew, xold, rows):
    """np.interp of every row of rows (..., len(xold)) at xnew."""
    flat = rows.reshape(-1, rows.shape[-1])
    out = np.stack([np.interp(xnew, xold, r) for r in flat])
    return out.reshape(rows.shape[:-1] + (xnew.size,))


def main(fpath, ppath):
    F = h5py.File(fpath, "r")
    P = h5py.File(ppath, "r")
    hdr = F["header"].attrs
    S, Np = int(hdr["n_modes"]), int(hdr["n_particles"])
    if S == 0 or "yi" not in P:
        sys.exit("both files must come from a DO run (S > 0)")
    fs, ps = F["step"][:], P["step"][:]
    common = sorted(set(fs[fs >= 0].tolist()) & set(ps.tolist()))
    if not common:
        sys.exit("no common step between the two files")
    fi = np.array([int(np.flatnonzero(fs == s)[0]) for s in common])
    pi = np.array([int(np.flatnonzero(ps == s)[0]) for s in common])
    print(f"{fpath}\n{ppath}\n{len(common)} common instants, steps {common[0]}..{common[-1]}"
          f" (filter rows {len(fs[fs >= 0])}, python rows {len(ps)}); S = {S}, Np = {Np}")

    rows = []

    def add(name, a, b, note=""):
        a, b = np.asarray(a, float), np.asarray(b, float)
        rows.append((name, mx(a - b), mx(a), note))

    # ---- the solver state both routes copy
    Y = P["yi"][:][pi]                                               # (T, Np, S)
    add("Y_1", F["Y1_p"][:][fi], Y[:, :, 0], "same solver numbers, float32 rounding")

    # ---- area inner products
    A0p, Aip, ap, Gp, n2p = (P[k][:][pi] for k in ("A0", "Ai", "a", "G", "mean_norm2"))
    add("A0 = <ubar, phi>", F["A0"][:][fi], A0p)
    add("A_i = <u_i, phi>", F["Ai"][:][fi], Aip)
    add("a_i = <ubar, u_i>", F["a"][:][fi], ap)
    add("G_ij = <u_i, u_j>", F["G"][:][fi], Gp)
    add("<ubar, ubar>", F["mean_norm2"][:][fi], n2p, "plane: rectangle rule on the x = -5 and x = 40 columns")
    add("A_p", F["A_p"][:][fi], A0p[:, None] + np.einsum("tps,ts->tp", Y, Aip))
    add("M1_p = a_1 + Y_1", F["M1_p"][:][fi], ap[:, 0][:, None] + Y[:, :, 0])

    # ---- centreline (python: 801 uniform points, trapezoid; filter: edge GLL points, exact)
    xp = P["x"][0, :]
    xa = xp[xp >= 0]
    vm = P["v_axis_mean"][:][pi].astype(float)                       # (T, 801)
    vmo = P["v_axis_modes"][:][pi].astype(float)                     # (T, S, 801)
    w = np.full(xa.size, xa[1] - xa[0])
    w[[0, -1]] *= 0.5
    c = np.einsum("ti,i,ti->t", vm, w, vm)
    b = np.einsum("ti,i,tsi->ts", vm, w, vmo)
    C = np.einsum("tsi,i,tri->tsr", vmo, w, vmo)
    add("c = int vbar^2 dx", F["c"][:][fi], c, "plane: trapezoid on 801 points")
    add("b_i = int vbar v_i dx", F["b"][:][fi], b)
    add("C_ij = int v_i v_j dx", F["C"][:][fi], C)
    i2 = int(np.argmin(np.abs(xa - 2.0)))
    add("vbar(2, 0)", F["v2_mean"][:][fi], vm[:, i2])
    add("v_i(2, 0)", F["v2_modes"][:][fi], vmo[:, :, i2])
    q = c[:, None] + 2 * np.einsum("tps,ts->tp", Y, b) + np.einsum("tps,tsr,tpr->tp", Y, C, Y)
    v2 = vm[:, i2][:, None] + np.einsum("tps,ts->tp", Y, vmo[:, :, i2])
    add("M_p", F["M_p"][:][fi], np.sign(v2) * np.sqrt(np.maximum(q, 0.0)))

    # ---- the lines, python interpolated to the filter's points
    ls = F["line_step"][:]
    lcommon = [s for s in common if s in set(ls.tolist())]
    li = np.array([int(np.flatnonzero(ls == s)[0]) for s in lcommon])
    lp = np.array([int(np.flatnonzero(ps == s)[0]) for s in lcommon])
    if lcommon:
        xax = F["x_axis"][:]
        va = F["v_axis"][:][li]                                      # (TL, S+1, nA)
        add("vbar on the centreline", va[:, 0], interp_rows(xax, xa, P["v_axis_mean"][:][lp].astype(float)),
            f"{len(lcommon)} line instants; python linearly interpolated to the GLL points")
        add("v_i on the centreline", va[:, 1:], interp_rows(xax, xa, P["v_axis_modes"][:][lp].astype(float)))
        xw = F["x_wall"][:]                                          # (2, nW)
        pw = F["p_wall"][:][li]                                      # (TL, S+1, 2, nW)
        pwm = P["p_wall_mean"][:][lp].astype(float)                  # (TL, 2, 801)
        pwo = P["p_wall_modes"][:][lp].astype(float)                 # (TL, S, 2, 801)
        for wl, name in ((0, "lower"), (1, "upper")):
            add(f"pbar on the {name} wall", pw[:, 0, wl], interp_rows(xw[wl], xa, pwm[:, wl]))
            add(f"p_i on the {name} wall", pw[:, 1:, wl], interp_rows(xw[wl], xa, pwo[:, :, wl]))

    # ---- Delta L, python rule (last u < 0 on |y| = 1.475) on the near-wall lines
    xwall = xp[xp > 0]
    uwm = P["u_wall_mean"][:][pi].astype(float)                      # (T, 2, 800)
    uwo = P["u_wall_modes"][:][pi].astype(float)                     # (T, S, 2, 800)
    dLf = F["dL_p"][:][fi].astype(float)
    dLu = np.empty_like(dLf)
    for p0 in range(0, Np, CHUNK):
        Yc = Y[:, p0:p0 + CHUNK]
        uw = uwm[:, None] + np.einsum("tps,tscx->tpcx", Yc, uwo)    # (T, chunk, 2, 800)
        dLu[:, p0:p0 + CHUNK] = last_negative(uw[:, :, 1], xwall) - last_negative(uw[:, :, 0], xwall)
    add("dL_p vs dL (last u < 0 at |y| = 1.475)", dLf, dLu, "different rules, see the notes")
    within = float(np.mean(np.abs(dLf - dLu) <= 0.1))

    # ---- from the kept planes: y_b and the wall shear
    notes_plane = ""
    if "grid" in P:
        gs = P["grid/step"][:]
        gcommon = [s for s in common if s in set(gs.tolist())]
        gi = np.array([int(np.flatnonzero(gs == s)[0]) for s in gcommon])
        ci = np.array([common.index(s) for s in gcommon])
        if gcommon:
            yrow = P["y"][:, 0]
            dx, dy = xp[1] - xp[0], yrow[1] - yrow[0]
            jlow, jup = 1, yrow.size - 2
            cols = xp >= 0
            # rectangle weights of the plane over x >= 0 (the python convention)
            Wr = np.full((yrow.size, int(cols.sum())), dx * dy)
            # the same with trapezoid edges (half weight on the boundary rows and columns)
            Wt = Wr.copy()
            Wt[[0, -1], :] *= 0.5
            Wt[:, [0, -1]] *= 0.5
            ybf = F["yb_p"][:][fi][ci].astype(float)
            dudnf = F["dudn_wall"][:]
            lsel = [k for k, s in enumerate(gcommon) if s in set(ls.tolist())]
            ybr, ybt, taud = [], [], []
            for k, g in enumerate(gi):
                mu = P["grid/mean_u"][g].astype(float)                # (NY, NX)
                mou = P["grid/mode_u"][g].astype(float)               # (S, NY, NX)
                Yk = Y[ci[k]]                                         # (Np, S)
                for W, acc in ((Wr, ybr), (Wt, ybt)):
                    I1, Iy = W.sum(), (W * yrow[:, None]).sum()
                    um1, umy = (W * mu[:, cols]).sum(), (W * yrow[:, None] * mu[:, cols]).sum()
                    ui1 = np.einsum("yx,syx->s", W, mou[:, :, cols])
                    uiy = np.einsum("yx,syx->s", W * yrow[:, None], mou[:, :, cols])
                    acc.append((Iy - umy - Yk @ uiy) / (I1 - um1 - Yk @ ui1))
                if k in lsel:
                    j = int(np.flatnonzero(ls == gcommon[k])[0])
                    tm = wall_dudn(mu, dy, jlow, jup)[:, cols]        # (2, 801)
                    to = wall_dudn(mou, dy, jlow, jup)[:, :, cols]    # (S, 2, 801)
                    xw = F["x_wall"][:]
                    for wl in (0, 1):
                        taud.append((dudnf[j, 0, wl] - np.interp(xw[wl], xa, tm[wl]),
                                     dudnf[j, 1:, wl] - interp_rows(xw[wl], xa, to[:, wl]),
                                     dudnf[j, 0, wl], dudnf[j, 1:, wl]))
            ybr, ybt = np.array(ybr), np.array(ybt)
            add("yb_p vs plane, rectangle rule", ybf, ybr, f"{len(gcommon)} plane instants")
            add("yb_p vs plane, trapezoid edges", ybf, ybt)
            # the area inner products again, with trapezoid edges on the whole plane
            Wg = trap_weights(xp, yrow)
            Gt, n2t, at = [], [], []
            for g in gi:
                mu, mv = P["grid/mean_u"][g].astype(float), P["grid/mean_v"][g].astype(float)
                mou, mov = P["grid/mode_u"][g].astype(float), P["grid/mode_v"][g].astype(float)
                Gt.append(np.einsum("syx,yx,ryx->sr", mou, Wg, mou) + np.einsum("syx,yx,ryx->sr", mov, Wg, mov))
                n2t.append((Wg * (mu * mu + mv * mv)).sum())
                at.append(np.einsum("yx,yx,syx->s", mu, Wg, mou) + np.einsum("yx,yx,syx->s", mv, Wg, mov))
            add("G_ij, plane with trapezoid edges", F["G"][:][fi][ci], np.array(Gt))
            add("<ubar, ubar>, plane with trapezoid edges", F["mean_norm2"][:][fi][ci], np.array(n2t))
            add("a_i, plane with trapezoid edges", F["a"][:][fi][ci], np.array(at))
            if taud:
                rows.append(("dubar/dn on the walls", mx([d[0] for d in taud]), mx([d[2] for d in taud]),
                             "python: central difference at |y| = 1.475; filter: exact on the wall"))
                rows.append(("du_i/dn on the walls", mx([d[1] for d in taud]), mx([d[3] for d in taud]), ""))
            # Delta L with the python shear rule (dLs) on the planes
            dLs = np.empty((len(gcommon), Np))
            for k, g in enumerate(gi):
                mu = P["grid/mean_u"][g].astype(float)
                mou = P["grid/mode_u"][g].astype(float)
                tm = wall_dudn(mu, dy, jlow, jup)[:, xp > 0]            # (2, 800)
                to = wall_dudn(mou, dy, jlow, jup)[:, :, xp > 0]        # (S, 2, 800)
                for p0 in range(0, Np, CHUNK):
                    tw = tm[None] + np.einsum("ps,scx->pcx", Y[ci[k], p0:p0 + CHUNK], to)
                    dLs[k, p0:p0 + CHUNK] = last_negative(tw[:, 1], xwall) - last_negative(tw[:, 0], xwall)
            add("dL_p vs dLs (last du/dn < 0, central difference)", F["dL_p"][:][fi][ci], dLs)
            notes_plane = (f"  fraction of (instant, particle) with |dL_p - dLs| <= 0.1: "
                           f"{float(np.mean(np.abs(F['dL_p'][:][fi][ci] - dLs) <= 0.1)):.3f}")
        else:
            notes_plane = "  no plane instant in common: y_b and the wall shear not compared"
    else:
        notes_plane = "  python file without grid/: y_b and the wall shear not compared (rerun with KEEP_EVERY 1)"

    # ---- table
    wname = max(len(r[0]) for r in rows)
    print(f"\n{'quantity':{wname}}  {'max |filter - python|':>22}  {'max |filter|':>13}  note")
    for name, d, s, note in rows:
        print(f"{name:{wname}}  {d:22.3e}  {s:13.4e}  {note}")
    print(f"\n  fraction of (instant, particle) with |dL_p - dL| <= 0.1: {within:.3f}")
    print(notes_plane)
    print("  A, M, M1, a, G, c, b, C, v(2, 0), Y_1: same definition on both sides, the difference is the\n"
          "  plane's quadrature/interpolation error. dL, dLs, du/dn: different definitions (docstring).")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
