import sys
import numpy as np, h5py
import matplotlib.pyplot as plt
sys.path.insert(0, "../../cases/se/templates")
from extract_lib import ip

phi = np.load("../../cases/se/templates/phi_Re100.npz")
phi_u, phi_v = phi["phi_u"], phi["phi_v"]

run = "../../cases/se/runs/2026_09_13_jed_sweep/Re100_sig0.0144/do_Re100_sig0.0144_S6_Np1000_seed45_eps0.0001"
with h5py.File(run + "/output/extract/measures.h5", "r") as h:
    t = h["grid/t"][:]
    u1, v1 = h["grid/mode_u"][:, 0], h["grid/mode_v"][:, 0]

ru, rv = u1 - phi_u, v1 - phi_v
res = np.sqrt([ip(a, b, a, b) for a, b in zip(ru, rv)])

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
plt.plot(t, res, "-", lw=0.8, color="#0072B2")
plt.xlabel("$t$"); plt.ylabel(r"$\|u_1 - \phi\|$")
plt.show()
