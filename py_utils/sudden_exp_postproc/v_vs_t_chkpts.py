import numpy as np
from do_quicklook import read_his
import matplotlib.pyplot as plt

t, pts = read_his("/home/isma/nektar_src_full/cases/se/runs/2026_07_15_do_clean_dnspod_sig0p0144_t250/output/asymmetry/axis_v.his")
v = pts[:, :, 1]
x = 0.1 * np.arange(v.shape[1])
M = np.sign(v[:, 20]) * np.sqrt(np.trapezoid(v**2, x, axis=1))
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(t, M, label=f"M")
for xx in (1, 2, 5, 10):
    ax.plot(t, pts[:, int(xx/0.1), 1], label=f"v at x={xx}", lw=0.8)
ax.legend()
ax.set_xlabel("t")
plt.show()
