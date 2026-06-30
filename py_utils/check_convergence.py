#!/usr/bin/env python3
"""
Check convergence/settling from velocity data stored in NPZ/HDF5.

Input container is expected to contain:
    - u: (nt, ny, nx)
    - v: (nt, ny, nx)
    - t: (nt,)

This script computes:
    - relative L2 snapshot change for u and v
    - RMS velocity increment between snapshots
    - domain-mean kinetic energy time series

It then reports:
    - steady_converged      (residuals are small in tail)
    - periodic_settled      (periodic amplitude/frequency stable in tail)
    - not_converged
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from py_utils.common.data_io import close_data, load_data, read_array
except Exception:  # pragma: no cover - direct script execution fallback
    from common.data_io import close_data, load_data, read_array


def _validate(u: np.ndarray, v: np.ndarray, t: np.ndarray) -> None:
    if u.ndim != 3 or v.ndim != 3:
        raise ValueError(f"u and v must be 3D: got u={u.shape}, v={v.shape}")
    if u.shape != v.shape:
        raise ValueError(f"u and v shapes must match: u={u.shape}, v={v.shape}")
    if t.ndim != 1 or t.shape[0] != u.shape[0]:
        raise ValueError(f"t must be (nt,), got t={t.shape}, nt={u.shape[0]}")
    if u.shape[0] < 4:
        raise ValueError("Need at least 4 snapshots for convergence check.")


def _coerce_time_first(u: np.ndarray, v: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder velocity arrays so that time is axis 0.

    Supports containers where u,v are either (nt, ny, nx) or (ny, nx, nt).
    """
    nt = int(t.shape[0])
    if u.shape[0] == nt and v.shape[0] == nt:
        return u, v

    u_axes = [ax for ax, n in enumerate(u.shape) if n == nt]
    v_axes = [ax for ax, n in enumerate(v.shape) if n == nt]
    if len(u_axes) != 1 or len(v_axes) != 1:
        raise ValueError(
            "Cannot infer time axis uniquely for u/v. "
            f"u shape={u.shape}, v shape={v.shape}, t shape={t.shape}."
        )

    u_tf = np.moveaxis(u, u_axes[0], 0)
    v_tf = np.moveaxis(v, v_axes[0], 0)
    return u_tf, v_tf


def _safe_rel(num: np.ndarray, den: np.ndarray, eps: float = 1.0e-30) -> np.ndarray:
    return num / np.maximum(den, eps)


def _dominant_frequency(signal: np.ndarray, dt: float) -> float:
    sig = np.asarray(signal, dtype=float)
    n = sig.size
    if n < 8 or dt <= 0:
        return 0.0
    sig = sig - sig.mean()
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=dt)
    if freqs.size <= 1:
        return 0.0
    mag = np.abs(spec)
    mag[0] = 0.0
    k = int(np.argmax(mag))
    return float(freqs[k])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Input velocity NPZ/HDF5. Defaults to ./out.h5 when omitted.",
    )
    ap.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Deprecated alias for --data (kept for backward compatibility).",
    )
    ap.add_argument(
        "--mode",
        choices=["auto", "steady", "periodic"],
        default="auto",
        help="Expected regime. auto tries steady first, then periodic.",
    )
    ap.add_argument(
        "--tail-frac",
        type=float,
        default=0.4,
        help="Fraction of final samples used for tail-based checks.",
    )
    ap.add_argument(
        "--steady-rel-thresh",
        type=float,
        default=1.0e-4,
        help="Tail median threshold for relative L2 change (steady check).",
    )
    ap.add_argument(
        "--steady-rms-thresh",
        type=float,
        default=1.0e-4,
        help="Tail median threshold for RMS velocity increment (steady check).",
    )
    ap.add_argument(
        "--periodic-amp-tol",
        type=float,
        default=0.10,
        help="Allowed relative change in oscillation amplitude between two tail halves.",
    )
    ap.add_argument(
        "--periodic-freq-tol",
        type=float,
        default=0.05,
        help="Allowed relative change in dominant frequency between two tail halves.",
    )
    ap.add_argument(
        "--periodic-min-amp-ratio",
        type=float,
        default=1.0e-4,
        help="Minimum tail oscillation amplitude ratio to consider periodic behavior real.",
    )
    ap.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV for per-step metrics.",
    )
    ap.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional PNG plot of key metrics.",
    )
    args = ap.parse_args()

    if not (0.1 <= args.tail_frac <= 0.9):
        raise ValueError("--tail-frac must be between 0.1 and 0.9")

    if args.data is not None and args.npz is not None:
        raise ValueError("Pass only one of --data or --npz.")

    in_path = args.data if args.data is not None else args.npz
    if in_path is None:
        in_path = Path("out.h5")

    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path}. Pass --data <path> or --npz <path>."
        )

    data = load_data(in_path)
    try:
        u = read_array(data, "u")
        v = read_array(data, "v")
        t = read_array(data, "t")
    finally:
        close_data(data)

    u, v = _coerce_time_first(u, v, t)
    _validate(u, v, t)
    nt = u.shape[0]

    du = u[1:] - u[:-1]
    dv = v[1:] - v[:-1]

    du_l2 = np.sqrt(np.mean(du * du, axis=(1, 2)))
    dv_l2 = np.sqrt(np.mean(dv * dv, axis=(1, 2)))
    u_l2 = np.sqrt(np.mean(u[1:] * u[1:], axis=(1, 2)))
    v_l2 = np.sqrt(np.mean(v[1:] * v[1:], axis=(1, 2)))
    rel_u = _safe_rel(du_l2, u_l2)
    rel_v = _safe_rel(dv_l2, v_l2)
    rms_delta = np.sqrt(np.mean(du * du + dv * dv, axis=(1, 2)))
    step_time = t[1:]

    ke = 0.5 * np.mean(u * u + v * v, axis=(1, 2))

    n_tail = max(8, int(round((nt - 1) * args.tail_frac)))
    n_tail = min(n_tail, nt - 1)
    tail_slice = slice((nt - 1) - n_tail, nt - 1)

    rel_u_tail_med = float(np.median(rel_u[tail_slice]))
    rel_v_tail_med = float(np.median(rel_v[tail_slice]))
    rms_tail_med = float(np.median(rms_delta[tail_slice]))

    steady_ok = (
        rel_u_tail_med <= args.steady_rel_thresh
        and rel_v_tail_med <= args.steady_rel_thresh
        and rms_tail_med <= args.steady_rms_thresh
    )

    dt_tail = np.diff(t)
    dt_med = float(np.median(dt_tail)) if dt_tail.size else 1.0
    if dt_med <= 0:
        dt_med = 1.0

    n_tail_ke = max(10, int(round(nt * args.tail_frac)))
    n_tail_ke = min(n_tail_ke, nt)
    ke_tail = ke[-n_tail_ke:]
    half = n_tail_ke // 2
    ke_a = ke_tail[:half]
    ke_b = ke_tail[half:]

    if ke_a.size >= 8 and ke_b.size >= 8:
        amp_a = float(0.5 * (np.max(ke_a) - np.min(ke_a)))
        amp_b = float(0.5 * (np.max(ke_b) - np.min(ke_b)))
        amp_ref = max(1.0e-30, 0.5 * (amp_a + amp_b))
        amp_rel_change = abs(amp_b - amp_a) / amp_ref

        freq_a = _dominant_frequency(ke_a, dt_med)
        freq_b = _dominant_frequency(ke_b, dt_med)
        freq_ref = max(1.0e-30, 0.5 * (abs(freq_a) + abs(freq_b)))
        freq_rel_change = abs(freq_b - freq_a) / freq_ref

        amp_ratio = amp_ref / max(abs(float(np.mean(ke_tail))), 1.0e-30)
    else:
        amp_a = amp_b = 0.0
        amp_rel_change = np.inf
        freq_a = freq_b = 0.0
        freq_rel_change = np.inf
        amp_ratio = 0.0

    periodic_ok = (
        amp_ratio >= args.periodic_min_amp_ratio
        and amp_rel_change <= args.periodic_amp_tol
        and freq_rel_change <= args.periodic_freq_tol
    )

    if args.mode == "steady":
        verdict = "steady_converged" if steady_ok else "not_converged"
    elif args.mode == "periodic":
        verdict = "periodic_settled" if periodic_ok else "not_converged"
    else:
        if steady_ok:
            verdict = "steady_converged"
        elif periodic_ok:
            verdict = "periodic_settled"
        else:
            verdict = "not_converged"

    summary = {
        "verdict": verdict,
        "mode": args.mode,
        "nt": int(nt),
        "tail_frac": float(args.tail_frac),
        "steady": {
            "rel_u_tail_median": rel_u_tail_med,
            "rel_v_tail_median": rel_v_tail_med,
            "rms_delta_tail_median": rms_tail_med,
            "rel_thresh": float(args.steady_rel_thresh),
            "rms_thresh": float(args.steady_rms_thresh),
            "steady_ok": bool(steady_ok),
        },
        "periodic": {
            "amp_a": amp_a,
            "amp_b": amp_b,
            "amp_rel_change": float(amp_rel_change),
            "freq_a": float(freq_a),
            "freq_b": float(freq_b),
            "freq_rel_change": float(freq_rel_change),
            "amp_ratio": float(amp_ratio),
            "amp_tol": float(args.periodic_amp_tol),
            "freq_tol": float(args.periodic_freq_tol),
            "min_amp_ratio": float(args.periodic_min_amp_ratio),
            "periodic_ok": bool(periodic_ok),
        },
    }

    print(json.dumps(summary, indent=2))

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["time", "rel_u", "rel_v", "rms_delta"])
            for i in range(step_time.size):
                w.writerow(
                    [
                        f"{step_time[i]:.12g}",
                        f"{rel_u[i]:.12g}",
                        f"{rel_v[i]:.12g}",
                        f"{rms_delta[i]:.12g}",
                    ]
                )
        print(f"metrics_csv: {args.csv_out}")

    if args.plot_out is not None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(step_time, rel_u, label="rel_u")
        ax[0].plot(step_time, rel_v, label="rel_v")
        ax[0].axhline(args.steady_rel_thresh, color="k", linestyle="--", linewidth=1)
        ax[0].set_yscale("log")
        ax[0].set_ylabel("relative L2")
        ax[0].legend(loc="best")

        ax[1].plot(step_time, rms_delta, label="rms_delta")
        ax[1].axhline(args.steady_rms_thresh, color="k", linestyle="--", linewidth=1)
        ax[1].set_yscale("log")
        ax[1].set_ylabel("RMS delta")
        ax[1].legend(loc="best")

        ax[2].plot(t, ke, label="mean kinetic energy")
        ax[2].set_ylabel("KE")
        ax[2].set_xlabel("time")
        ax[2].legend(loc="best")
        fig.suptitle(f"Convergence check: {verdict}")
        fig.tight_layout()

        args.plot_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_out, dpi=150)
        plt.close(fig)
        print(f"plot_png: {args.plot_out}")

    return 0 if verdict != "not_converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
