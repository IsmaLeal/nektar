# Figures of the sudden-expansion campaign. One folder per set of parameters, so figures of
# different Re, sigma, S, Np or eps never share a folder:
#
#   Re<Re>/                 unforced, one Re: phi, the adjoint mode, eigenvalue fits, kicks
#   Re<Re>_sig<sigma>/      one forced pair, as in cases/se/runs/2026_09_13_jed_sweep/:
#     <run>/                one run, the folder named as its run directory (crossings, movies, phi_u1;
#                           the twin figure sits in the folder of the DNS twin)
#     vary_<p>_<fixed>/     one line of DO runs, which differ in p only, each figure with the DNS (histograms)
#     *                     figures of all the runs of the pair (burn-in, q-dagger projection)
#   sweep/                  figures across Re or across (Re, sigma) pairs
#   candidacy/              the figure set of the candidacy plan and slides, copied out with 'make candidacy'
#
#   Names: the folders hold the parameters, a name is the quantity plus what its folder leaves
#   open, no dates and no figure numbers (the date is in git):
#     Re100/phi.pdf, Re100_sig0.0144/vary_S_Np1000_eps1e-4/hist_A.png,
#     Re100_sig0.0144/do_Re100_sig0.0144_S6_Np1000_seed45_eps0.0001/cross_p269.png
#   Every script takes --out and writes <figure>.cmd next to the figure (figsave.py)
#   with the command line that made it. Diagnostics that only make sense with one
#   run (stats_*, do_panels_*, burn-in checks) stay in that run's output/. Movies are not in git.
#
# Targets regenerate from the local extraction (cases/se/runs/2026_09_13_jed_sweep):
#   make fields hists stills traces rates    make candidacy CAND=~/.../candidacy_exam

REPO  := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
PP    := $(REPO)/py_utils/sudden_exp_postproc
TPL   := $(REPO)/cases/se/templates
SWEEP := $(REPO)/cases/se/runs/2026_09_13_jed_sweep
PAIR  := $(SWEEP)/Re100_sig0.0144
CAND  ?= $(HOME)/todisimo/studies/7.0_epfl/year1/academic/candidacy_exam
PY    := python3

.PHONY: all fields hists stills traces rates candidacy

all: fields hists stills traces rates

# Every field figure comes from plot_plane.py, so they all share one x window (-5..15; --full = whole domain).
ADJ := $(REPO)/cases/se/runs/2026_07_12_m03_Re79p3_arnoldi_adjoint

fields: $(patsubst %,Re%/phi.pdf,90 100 110) Re79p3/qdagger.pdf

Re%/phi.pdf: $(TPL)/phi_Re%.npz $(PP)/plot_plane.py
	$(PY) $(PP)/plot_plane.py $< --title 'Re $*' --out $@

# leading adjoint eigenmode of the Re 79.3 Arnoldi run (geometry_eig_0.fld) on the extraction plane
$(TPL)/qdagger_Re79p3.npz: $(ADJ)/geometry_eig_0.fld $(PP)/plane_from_fld.py
	$(PY) $(PP)/plane_from_fld.py $< --geometry $(ADJ)/geometry.xml --out $@

Re79p3/qdagger.pdf: $(TPL)/qdagger_Re79p3.npz $(PP)/plot_plane.py
	$(PY) $(PP)/plot_plane.py $< --keys u v --labels '$$q^\dagger_u$$' '$$q^\dagger_v$$' --title 'Re 79.3' --out $@

# traces come from panel.py: every --panel is one axis, a plot kind (draw_<kind>.py) and the
# runs drawn in it (names as in se.py: the part after dns_<pair>_ / do_<pair>_, one argument = one curve).
PANEL := $(PY) $(PP)/panel.py --pair $(PAIR)
TAG   := $(notdir $(PAIR))

# One histogram per (quantity, line) in the line's folder, every line found by vary_hists.py: the DO runs of a
# pair that differ in one of S, Np, eps (vary_<that one>_<the rest>; forcing-seed repeats _frc pooled into their
# run), each figure with the DNS of the pair (seeds pooled, twins left out). 'python3 vary_hists.py --list'.
hists:
	$(PY) $(PP)/vary_hists.py
	$(PY) $(PP)/grid_hists.py
	$(PY) $(PP)/sweep_hists.py --out sweep

# phi and mode 1 of every DO run, manim house style (do_fields.py; the drawing needs the divulg venv)
stills:
	$(PY) $(PP)/do_fields.py

# Crossings: A(t) in two windows, for DNS seeds 45 and 47 and for four particles of one DO run
# (269 510 635 848: the draw of numpy default_rng(0) among its 1000). Twins: the DNS twin of
# particle 0 against that particle in the eps 1e-4 and eps 1e-8 DO runs of the same (S, Np).
traces:
	@set -e; for s in 45 47; do \
	    $(PANEL) --panel cross tlim=0:4000 seed$$s --panel cross tlim=4000:8000 \
	        --out $(TAG)/dns_$(TAG)_seed$$s/cross.png; \
	done
	@set -e; for p in 269 510 635 848; do \
	    $(PANEL) --panel cross tlim=0:4000 S6_Np1000_seed45_eps1e-4@$$p --panel cross tlim=4000:8000 \
	        --out $(TAG)/do_$(TAG)_S6_Np1000_seed45_eps0.0001/cross_p$$p.png; \
	done
	@set -e; for sn in "8 1000" "6 5000" "12 5000"; do set -- $$sn; \
	    $(PANEL) --sharex --tags \
	        --panel trace q=A seed45_Np$$2_S$$1p0 S$$1_Np$$2_seed45_eps1e-4@0 S$$1_Np$$2_seed45_eps1e-8@0 \
	        --panel trace q=M --panel trace q=dL --panel trace q=Y1 \
	        --out $(TAG)/dns_$(TAG)_seed45_Np$$2_S$$1p0/twin.pdf; \
	done

rates:
	mkdir -p sweep
	$(PY) $(PP)/sweep_rates.py --sweep $(SWEEP) --out sweep/sweep_rates

# The candidacy figure set: cases/se/figures/candidacy/ is the source, the tex and
# slides folders receive copies (never deleted, the folders also hold other files).
CAND_FIGS := convergence cov_spectrum do_trajectories injection_gain lambda_vs_Re mc_closure \
             ou_autocorr pdf_m rank_invariance se_doublewell_mt se_states transition_reconstructed \
             vortex_mowlavi_coeffs vortex_mowlavi_fields cpu_scaling cpu_scaling_time

candidacy:
	@for n in $(CAND_FIGS); do for e in pdf png; do \
	    [ -f candidacy/$$n.$$e ] && cp -v candidacy/$$n.$$e $(CAND)/tex/figures/ && cp candidacy/$$n.$$e $(CAND)/slides/figures/; \
	done; done; true
