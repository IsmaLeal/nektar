#!/usr/bin/env python3
"""Estimate vortex-shedding frequency from a centerline probe v(t).

Reads u/v/x/y/t from an out.h5 (load_chk.py output), takes a single
grid point closest to (--probe-x, --probe-y) — by default just
downstream of the cylinder at y=0 — and analyses the time series via
short-time FFT to track the dominant frequency over time.

Default probe variable is v: at y=0, by mean-flow symmetry <v>=0, so
the signal is pure shedding fluctuation oscillating at the FUNDAMENTAL
Strouhal frequency (each alternating-sign shed vortex pushes fluid
through y=0 with alternating sign). u at y=0 oscillates at 2× St
because both upper- and lower-shed vortices contribute symmetrically;
vorticity is a derived (noisier) quantity.

Outputs a PNG with two stacked panels:
  - top: probe signal vs time
  - bottom: spectrogram + dominant-frequency curve, with mean
            Strouhal over the second half annotated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="out.h5 (or out_merged.h5) from load_chk.py")
    ap.add_argument("--out",  type=Path, required=True,
                    help="Output PNG path")
    ap.add_argument("--var",  choices=["v", "u"], default="v",
                    help="Probe variable (default v; see header).")
    ap.add_argument("--probe-x", type=float, default=2.0,
                    help="Probe x location, default 2 (4 diameters downstream).")
    ap.add_argument("--probe-y", type=float, default=0.0,
                    help="Probe y location, default 0 (centerline).")
    ap.add_argument("--window-sec", type=float, default=10.0,
                    help="STFT window length in seconds (default 10).")
    ap.add_argument("--stride-sec", type=float, default=1.0,
                    help="STFT stride in seconds (default 1).")
    ap.add_argument("--fmax-plot", type=float, default=2.0,
                    help="Max frequency to display in spectrogram (default 2 Hz).")
    ap.add_argument("--U-inf", type=float, default=1.0,
                    help="Free-stream velocity for Strouhal (default 1).")
    ap.add_argument("--D", type=float, default=1.0,
                    help="Cylinder diameter for Strouhal (default 1).")
    args = ap.parse_args()

    with h5py.File(args.data, "r") as f:
        x = np.asarray(f["x"])              # (Ny, Nx)
        y = np.asarray(f["y"])              # (Ny, Nx)
        t = np.asarray(f["t"], dtype=float) # (T,)
        sig_field = np.asarray(f[args.var]) # (Ny, Nx, T)

    # Closest grid point to (probe-x, probe-y)
    ix = int(np.argmin(np.abs(x[0, :] - args.probe_x)))
    iy = int(np.argmin(np.abs(y[:, 0] - args.probe_y)))
    actual_x = float(x[iy, ix])
    actual_y = float(y[iy, ix])
    sig = sig_field[iy, ix, :]              # (T,)

    # Time spacing (chk-snapshot spacing). Assert uniform.
    dts = np.diff(t)
    dt = float(np.median(dts))
    if dts.max() / dts.min() > 1.05:
        print(f"warning: non-uniform sampling (dt range {dts.min()}..{dts.max()}); "
              "STFT assumes uniform — results may be biased near the irregular gaps.")

    # STFT params
    nperseg = max(16, int(round(args.window_sec / dt)))
    nstride = max(1,  int(round(args.stride_sec / dt)))
    nfft    = max(256, 1 << int(np.ceil(np.log2(nperseg))))   # zero-pad for finer freq resolution
    freqs   = np.fft.rfftfreq(nfft, d=dt)
    win     = np.hanning(nperseg)

    if len(t) < nperseg:
        print(f"error: signal length {len(t)} < window {nperseg} samples", file=sys.stderr)
        return 2

    centers, dom_freq, spec_cols = [], [], []
    for start in range(0, len(t) - nperseg + 1, nstride):
        seg = sig[start:start + nperseg]
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg * win, n=nfft))
        # Skip DC bin when picking the dominant freq
        idx_peak = 1 + int(np.argmax(spec[1:]))
        centers.append(0.5 * (t[start] + t[start + nperseg - 1]))
        dom_freq.append(freqs[idx_peak])
        spec_cols.append(spec)
    centers  = np.asarray(centers)
    dom_freq = np.asarray(dom_freq)
    S        = np.asarray(spec_cols).T   # (n_freq, n_window)

    # Strouhal: average over the second half of the trace (steady state)
    half = len(dom_freq) // 2
    St_curve = dom_freq * args.D / args.U_inf
    St_mean  = float(np.mean(St_curve[half:])) if half > 0 else float(np.mean(St_curve))

    # Plot
    fig, (ax_sig, ax_spec) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.4], hspace=0.08),
    )

    ax_sig.plot(t, sig, lw=0.7, color="#1f3b73")
    ax_sig.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_sig.set_ylabel(f"{args.var}(x={actual_x:.2f}, y={actual_y:.2f})")
    ax_sig.set_title(f"Cylinder wake shedding   probe ({args.var}@y=0, x={actual_x:.2f})   "
                     f"D={args.D}  U∞={args.U_inf}")
    ax_sig.grid(alpha=0.3)

    fmask = freqs <= args.fmax_plot
    pcm = ax_spec.pcolormesh(centers, freqs[fmask], S[fmask],
                             cmap="viridis", shading="auto")
    ax_spec.plot(centers, dom_freq, "r-", lw=1.6, label="dominant freq")
    ax_spec.set_xlim(t[0], t[-1])
    ax_spec.set_ylim(0, args.fmax_plot)
    ax_spec.set_xlabel("t")
    ax_spec.set_ylabel("frequency [1/time]")
    ax_spec.legend(loc="upper right")
    ax_spec.text(0.02, 0.95,
                 f"mean St (2nd half) = {St_mean:.4f}\n"
                 f"window={args.window_sec}s  stride={args.stride_sec}s  Δt={dt:.4g}s",
                 transform=ax_spec.transAxes, va="top", ha="left", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
    fig.colorbar(pcm, ax=ax_spec, label="|FFT| amplitude", pad=0.02)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")
    print(f"Probe: ({actual_x:.4f}, {actual_y:.4f}), variable={args.var}")
    print(f"Mean Strouhal (2nd half): {St_mean:.4f}")
    print(f"Mean dominant frequency (2nd half): {float(np.mean(dom_freq[half:])):.4f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
