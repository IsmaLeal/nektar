# 2026_08_08_dns_phi22p0_seed46 -- second independent DNS at phi = 22.0

Identical to 2026_08_08_dns_phi22p0_seed45 (see its README for mapping,
protocol, IC) except DOForcingSeed = 46 and FinTime = 10000 t.u.
Launched 2026-08-08 22:21, finished 2026-08-09 12:46 (~14.4 h,
predicted 14.9). NOTE: the pre-launch registration block was skipped on
a user interrupt; the registered expectation lived in chat: ~8-10
events at mean escape 970-1230 t.u. (band from seed45's 4-event
estimate).

## VERDICT (readout 2026-08-09 13:0x)

Paper criterion (c = 0.8, exit/reach thresholds, last-zero timestamps),
from axis_v.his -> M_of_t.npy:
  Mbar = 0.1696 (own deterministic attractor): 15 transitions
    at 485/1860/2282/3079/3149/3647/3743/4633/5097/5505/7307/7930/
    8459/8828/9941
  Mbar = 0.1935 (their WNL value):             11 transitions
15 exceeds the chat-registered 8-10 (P(N>=15|10.3) ~ 0.1, not a
violation; seed45 was low by the same luck).

POOLED ANCHOR (own-Mbar): seed45+46 = 19 events / 16000 t.u.
  = 1.19e-3 per t.u., mean escape ~840 t.u.
  Ducimetiere DNS at phi = 22: 13 / 1.1e4 = 1.18e-3 per t.u. -> ratio ~1.0.
  (Their-Mbar pooling: 15/16000 = 9.4e-4, mean ~1070 t.u.)
The two-point Kramers extrapolation (~1.7e-5 per t.u. at this sigma)
is refuted by both realisations independently.
