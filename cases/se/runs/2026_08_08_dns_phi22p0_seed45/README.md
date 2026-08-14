# 2026_08_08_dns_phi22p0_seed45 -- DNS at phi = 22.0 exactly (own eps)

Registered BEFORE launch, 2026-08-08.

Purpose: reference DNS inside the Ducimetiere et al. (PRF 9, 053905)
validated window (their Fig 4(d) density; their DNS anchor 13 transitions
per 1.1e4 t.u. at phi = 22), first leg of the DNS-then-DO comparison at
their forcing levels.

Mapping: phi = (sigma/eps)*sqrt(2*tau_zeta), tau_zeta = 25,
eps = 1/81.4 - 1/100 = 2.28501e-3 (own measured Re_c, three-instrument
collapse). sigma = 22.0*eps/sqrt(50) = 0.00710929. On THEIR axis
(eps = 1/79.3 - 1/100) the same run sits at phi = 19.3.

Protocol: identical to 2026_07_15_dns_clean_seed42 except
  sigma   0.00710929 (was 0.0144)
  seed    45 (42/43/44 used)
  FinTime 6000 (was 3000)
  chks    every 400 steps = 4 t.u. (was 2 t.u.); ~1500 chks ~ 3.5G
dt = 0.01, serial (DNS mode pins Np = 1), build_fast binary.
IC = fp_relax casefile_150.chk (unforced asymmetric fixed point, M = -0.1696),
i.e. the paper's start-at-the-attractor protocol (their eq 27); the
seed42/43 pair instead started from the symmetric base flow.

Registered expectation band (mean escape time at this sigma):
- Paper-anchor branch: their 13/1.1e4 t.u. at phi = 22 -> slow-time
  tau_esc = 2.21 -> /eps_own -> ~970 t.u. -> ~6 basin changes in 6000 t.u.
- Own two-point Kramers extrapolation (fit at sigma = 0.0144/0.0288;
  predates the crawl-vs-jump frame): 2.22e-2*exp(-3.72e-4/sigma^2)
  = 1.7e-5 per t.u. -> ~5.9e4 t.u. -> ~0 events in 6000 t.u.
Gates (hysteresis protocol c = 0.8, basin changes, NOT raw sign flips):
  >= 3 events  -> paper branch supported;
  0-1 events   -> consistent with the fit branch; EXTEND before concluding;
  2 events     -> ambiguous, extend.

Expected wall time ~26 h (15.7 s/t.u. serial).

## VERDICT (readout 2026-08-08 22:15, run completed 21:53)

4 basin changes in 6000 t.u. (times 823, 1834, 2899, 4918; protocol
c = 0.8, Mbar = 0.1696, from axis_v.his -> M_of_t.npy). Gate ">= 3 events"
PASSED: paper-anchor branch supported. Complete inter-event intervals
823/1011/1065/2019 -> mean ~1230 t.u., consistent with the registered
~970 t.u. (Poisson, 4 obs vs 6.2 expected); own two-point Kramers
extrapolation (~5.9e4 t.u., ~0 events) REFUTED at this sigma.
Forced well means -0.1915/+0.1926 (n = 2028/2344 t.u.) -- sit outward of
the unforced fixed point 0.1696, close to the paper's Mbar = 0.1935.
