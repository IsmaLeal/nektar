import argparse
import numpy as np
from pathlib import Path
from scipy.integrate import simpson
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 18})

ap = argparse.ArgumentParser()
ap.add_argument("--energy", type=Path, required=True, help="energy measures file.")
ap.add_argument("--asymmetry", type=Path, required=True, help="asymmetry measures file.")

args = ap.parse_args()

# asymmetry measure time series
raw = np.loadtxt(args.asymmetry)
npts = 201
d = raw.reshape(-1, npts, 4)
t = d[:, 0, 0]
v = d[:, :, 2]
x = np.linspace(0, 20, npts)
k2 = np.argmin(np.abs(x-2))
M = np.sign(v[:, k2]) * np.sqrt(simpson(v**2, x=x, axis=1))

# kinetic energy time series
raw = np.loadtxt(args.energy)
E = raw[:, 1]

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax2 = ax.twinx()

l1, = ax.plot(t, M, color="C0", label="Asymmetry measure")
l2, = ax2.plot(t, E, color="C1", label="Kinetic energy")
ax.set_xlabel("Time")
ax.set_ylabel("Kinetic energy", color="C0")
ax.set_yscale('log')
ax2.set_ylabel("Kinetic energy", color="C1")
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")

ax.legend(handles=[l1, l2])

plt.tight_layout()
plt.show()


